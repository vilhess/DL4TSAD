import torch 
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange  
from torch.nn.functional import gumbel_softmax
from einops import rearrange
import numpy as np 
import math
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        elif mode == 'transform':
            x = self._normalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
class ChannelMaskGenerator(nn.Module):
    def __init__(self, input_size, n_vars):
        super().__init__()
        self.generator = nn.Sequential(nn.Linear(input_size*2, n_vars, bias=False), nn.Sigmoid())
        with torch.no_grad():
            self.generator[0].weight.zero_()
        self.n_vars = n_vars

    def forward(self, x):
        distribution_matrix = self.generator(x)
        resample_matrix = self._bernoulli_gumbel_softmax(distribution_matrix)
        inverse_eye = 1 - torch.eye(self.n_vars).to(x.device)
        diag = torch.eye(self.n_vars).to(x.device)
        resample_matrix = torch.einsum('bcd,cd->bcd', resample_matrix, inverse_eye) + diag
        return resample_matrix

    def _bernoulli_gumbel_softmax(self, distribution_matrix):
        b, c, d = distribution_matrix.shape
        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix
        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)
        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)
        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class DynamicalContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, k=0.3):
        super().__init__()
        self.temperature = temperature
        self.k = k

    def forward(self, scores, attn_mask, norm_matrix):
        b = scores.shape[0]
        n_vars = scores.shape[-1]

        cosine = (scores / norm_matrix).mean(1)
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask

        all_scores = torch.exp(cosine / self.temperature)

        clustering_loss = -torch.log(pos_scores.sum(dim=-1) / all_scores.sum(dim=-1))

        eye = torch.eye(attn_mask.shape[-1]).unsqueeze(0).repeat(b, 1, 1).to(attn_mask.device)
        regular_loss = 1 / (n_vars * (n_vars - 1)) * torch.norm(eye.reshape(b, -1) - attn_mask.reshape((b, -1)),
                                                                p=1, dim=-1)
        loss = clustering_loss.mean(1) + self.k * regular_loss

        mean_loss = loss.mean()
        return mean_loss
class c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim_head = dim_head
        self.heads=heads
        self.d_k = math.sqrt(dim_head)
        inner_dim = dim_head*heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.dynamicalContranstiveLoss = DynamicalContrastiveLoss(k=regular_lambda, temperature=temperature)

    def forward(self, x, attn_mask=None):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        scale = 1/self.d_k

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        scores = einsum('b h i d, b h j d -> b h i j', q, k)

        dynamical_contrastive_loss = None

        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        norm_matrix = torch.einsum('bhid, bhjd -> bhij', q_norm, k_norm)

        if attn_mask is not None:
            def _mask (scores, attn_mask):
                large_negative = -math.log(1e10)
                attention_mask = torch.where(attn_mask==0, large_negative, 0)
                scores = scores*attn_mask.unsqueeze(1) + attention_mask.unsqueeze(1)
                return scores
            masked_scores = _mask(scores, attn_mask)
            dynamical_contrastive_loss = self.dynamicalContranstiveLoss(scores, attn_mask, norm_matrix)
        else:
            masked_scores = scores
        
        attn = self.attend(masked_scores*scale)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn, dynamical_contrastive_loss
class c_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, c_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, regular_lambda=regular_lambda, temperature=temperature)),
                    PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout))
                ])
            )
    
    def forward(self, x, attn_mask=None):
        total_loss=0
        for attn, ff in self.layers:
            x_n, attn, dcloss = attn(x, attn_mask=attn_mask)
            total_loss+=dcloss
            x = x_n + x
            x = ff(x) + x
        dcloss = total_loss / len(self.layers)
        return x, attn, dcloss
class Trans_c(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, d_model, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout, regular_lambda=regular_lambda, temperature=temperature)
        self.mlp_head = nn.Linear(dim, d_model)
    def forward(self, x, attn_mask=None):
        x = self.to_patch_embedding(x)
        x, attn, dcloss = self.transformer(x, attn_mask)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x, dcloss
class FlattenHead(nn.Module):
    def __init__(self, individual, n_vars, nf, seq_len, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, seq_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, seq_len)
            self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears1[i](z)
                z = self.dropouts[i](z)
                z = z.unsqueeze(1)
                x_out.append(z)
            x = torch.cat(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
        return x
class CATCHModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.revin_layer = RevIN(num_features=configs.in_dim, affine=configs.affine, subtract_last=configs.subtract_last)

        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.seq_len = configs.ws+1
        patch_num = int((self.seq_len - configs.patch_size) / configs.patch_stride + 1)

        self.norm = nn.LayerNorm(self.patch_size)
        self.re_attn=True

        self.mask_generator = ChannelMaskGenerator(input_size=configs.patch_size, n_vars=configs.in_dim)
        self.frequency_transformer = Trans_c(dim=configs.cf_dim, depth=configs.e_layers, heads=configs.n_heads, mlp_dim=configs.d_ff, dim_head=configs.head_dim, dropout=configs.dropout,
                                            patch_dim=configs.patch_size*2, d_model=configs.d_model*2, regular_lambda=configs.regular_lambda, temperature=configs.temperature)
        
        self.get_r = nn.Linear(configs.d_model * 2, configs.d_model * 2)
        self.get_i = nn.Linear(configs.d_model * 2, configs.d_model * 2)

        self.head_nf_f = configs.d_model * 2 * patch_num
        self.head_f1 = FlattenHead(configs.individual, configs.in_dim, self.head_nf_f, configs.ws+1, configs.head_dropout)
        self.head_f2 = FlattenHead(configs.individual, configs.in_dim, self.head_nf_f, configs.ws+1, configs.head_dropout)

        self.ircom = nn.Linear(self.seq_len*2, self.seq_len)
        
        
        
    def forward(self, z): # bs, ws, feat
        z = self.revin_layer(z, "norm")
        z = z.permute(0, 2, 1) # bs, feat, ws

        z = torch.fft.fft(z)
        z1 = z.real # bs, feat, ws
        z2 = z.imag # bs, feat, ws

        z1 = z1.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride) # bs, feat, patch_num, patch_size
        z2 = z2.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride) # bs, feat, patch_num, patch_size
        
        z1 = z1.permute(0, 2, 1, 3) # bs, patch_num, feat, patch_size
        z2 = z2.permute(0, 2, 1, 3) # bs, patch_num, feat, patch_size

        batch_size = z1.size(0)
        patch_num = z1.size(1)
        in_dim = z1.size(2)
        patch_size = z1.size(-1)

        z1 = torch.reshape(z1, (batch_size*patch_num, in_dim, patch_size)) # bs*patch_num, feat, patch_size
        z2 = torch.reshape(z2, (batch_size*patch_num, in_dim, patch_size)) # bs*patch_num, feat, patch_size

        z_cat = torch.cat((z1, z2), dim=-1) # bs*patch_num, feat, 2*patch_size

        channel_mask = self.mask_generator(z_cat) # bs*patch_num, feat, feat

        z, dcloss = self.frequency_transformer(z_cat, channel_mask) # bs*patch_num, feat, d_ff

        z1 = self.get_r(z) # bs*patch_num, feat, d_ff
        z2 = self.get_i(z) # bs*patch_num, feat, d_ff
    
        z1 = torch.reshape(z1, (batch_size, patch_num, in_dim, z1.shape[-1])) # bs, patch_num, feat, d_ff
        z2 = torch.reshape(z2, (batch_size, patch_num, in_dim, z2.shape[-1])) # bs, patch_num, feat, d_ff

        z1 = z1.permute(0, 2, 1, 3) # bs, feat, patch_num, d_ff
        z2 = z2.permute(0, 2, 1, 3) # bs, feat, patch_num, d_ff

        z1 = self.head_f1(z1)
        z2 = self.head_f2(z2)
        
        complex_z = torch.complex(z2, z2)
        z = torch.fft.ifft(complex_z)
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))
        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, "denorm")
        return z, complex_z.permute(0, 2, 1), dcloss
class frequency_loss(nn.Module):
    def __init__(self, configs, keep_dim=False, dim=None):
        super().__init__()
        self.keep_dim = keep_dim
        self.dim = dim
        self.fft = torch.fft.fft
        self.configs = configs

    def forward(self, outputs, batch_y):
        if not outputs.is_complex():
            outputs = self.fft(outputs, dim=1)
        loss_auxi = outputs - self.fft(batch_y, dim=1)
        loss_auxi = (loss_auxi.abs() ** 2).mean(dim=self.dim, keepdim=self.keep_dim)
        return loss_auxi
class frequency_criterion(torch.nn.Module):
    def __init__(self, configs):
        super(frequency_criterion, self).__init__()
        self.metric = frequency_loss(configs, dim=1, keep_dim=True)
        self.patch_size = configs.inference_patch_size
        self.patch_stride = configs.inference_patch_stride
        self.win_size = configs.ws +1
        self.patch_num = int((self.win_size - self.patch_size) / self.patch_stride + 1)
        self.padding_length = self.win_size - (self.patch_size + (self.patch_num - 1) * self.patch_stride)

    def forward(self, outputs, batch_y):

        output_patch = outputs.unfold(dimension=1, size=self.patch_size,
                                      step=self.patch_stride)

        b, n, c, p = output_patch.shape
        output_patch = rearrange(output_patch, 'b n c p -> (b n) p c')
        y_patch = batch_y.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        y_patch = rearrange(y_patch, 'b n c p -> (b n) p c')

        main_part_loss = self.metric(output_patch, y_patch)
        main_part_loss = main_part_loss.repeat(1, self.patch_size, 1)
        main_part_loss = rearrange(main_part_loss, '(b n) p c -> b n p c', b=b)

        end_point = self.patch_size + (self.patch_num - 1) * self.patch_stride - 1
        start_indices = np.array(range(0, end_point, self.patch_stride))
        end_indices = start_indices + self.patch_size

        indices = torch.tensor([range(start_indices[i], end_indices[i]) for i in range(n)]).unsqueeze(0).unsqueeze(-1)
        indices = indices.repeat(b, 1, 1, c).to(main_part_loss.device)
        main_loss = torch.zeros((b, n, self.win_size - self.padding_length, c)).to(main_part_loss.device)
        main_loss.scatter_(dim=2, index=indices, src=main_part_loss)

        non_zero_cnt = torch.count_nonzero(main_loss, dim=1)
        main_loss = main_loss.sum(1) / non_zero_cnt

        if self.padding_length > 0:
            padding_loss = self.metric(outputs[:, -self.padding_length:, :], batch_y[:, -self.padding_length:, :])
            padding_loss = padding_loss.repeat(1, self.padding_length, 1)
            total_loss = torch.concat([main_loss, padding_loss], dim=1)
        else:
            total_loss = main_loss
        return total_loss

import lightning as L

class CatchLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = CATCHModel(config)
        self.lr = config.lr
        self.Mlr = config.Mlr
        self.automatic_optimization = False
        self.criterion = nn.MSELoss()
        self.auxi_loss = frequency_loss(config)
        self.dc_lambda = config.dc_lambda
        self.auxi_lambda = config.auxi_lambda
        self.score_lambda = config.score_lambda
        self.len_loader = config.len_loader
        self.pct_start = config.pct_start
        self.num_epochs = config.epochs
        self.step = min(int(config.len_loader / 10), 100)
        self.temp_anomaly_score = nn.MSELoss(reduction="none")
        self.frequency_criterion = frequency_criterion(config)
    
    def training_step(self, batch, batch_idx):
        optim, optimM = self.optimizers()
        optim.zero_grad()
        x, _ = batch
        output, output_complex, dcloss = self.model(x)
        rec_loss = self.criterion(output, x)
        norm_input = self.model.revin_layer(x, "transform")
        auxi_loss = self.auxi_loss(output_complex, norm_input)
        loss = rec_loss + self.dc_lambda*dcloss + self.auxi_lambda*auxi_loss

        self.manual_backward(loss)
        if (batch_idx+1) % self.step == 0:
            optimM.step()
            optimM.zero_grad()
        optim.step()

    def configure_optimizers(self):
        main_params = [param for name, param in self.model.named_parameters() if 'mask_generator' not in name]
        optimizer = torch.optim.Adam(main_params, lr=self.lr)
        optimizerM = torch.optim.Adam(self.model.mask_generator.parameters(), lr=self.Mlr)
        return optimizer, optimizerM

    def get_loss(self, x, mode=None):
        outputs, _, _ = self.model(x)
        temp_score = torch.mean(self.temp_anomaly_score(outputs, x), dim=(1, 2))
        freq_score = torch.mean(self.frequency_criterion(outputs, x), dim=(1, 2))
        score = temp_score + self.score_lambda * freq_score
        return score


    def adjust_learning_rate(self, optimizer, epoch, printout=False):
        # lr = args.learning_rate * (0.2 ** (epoch // 2))
        lr_adjust = {epoch: self.lr * (0.5 ** ((epoch - 1) // 1))}
        
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if printout: print('Updating learning rate to {}'.format(lr))

    def on_train_epoch_end(self):
        optimizer, optimizerM = self.optimizers()
        self.adjust_learning_rate(optimizer, self.current_epoch+1, printout=True)
        self.adjust_learning_rate(optimizerM, self.current_epoch+1, printout=True)