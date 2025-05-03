import torch 
import torch.nn as nn 
from math import sqrt, log
from einops import rearrange, reduce, repeat
from tkinter import _flatten
import lightning as L 
from torchmetrics.classification import BinaryAUROC


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * - (log(10000)/d_model)).exp()
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 0::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
        
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dp=0.05):
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEncoding(d_model=d_model)
        self.dp = nn.Dropout(dp)

    def forward(self, x):
        x = self.value_embedding(x)
        x = x + self.position_embedding(x)
        return self.dp(x)
    
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 
        'cpu'
        )
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = torch.ones(self.num_features)
        self.affine_bias = torch.zeros(self.num_features)
        self.affine_weight=self.affine_weight.to(self.device)
        self.affine_bias=self.affine_bias.to(self.device)
        

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    

class DAC_Structure(nn.Module):
    def __init__(self, win_size, patch_size, channel, scale=None, attention_dp=0.05):
        super().__init__()

        self.scale=scale
        self.dp = nn.Dropout(attention_dp)
        self.window_size=win_size
        self.patch_size=patch_size
        self.channel=channel

    def forward(self, queries_patch_size, queries_patch_num, key_patch_size, key_patch_num, patch_index):

        # Inter-Patch
        B, L, H, E = queries_patch_size.shape # bs*channel, patch_num, nheads, d_model/nheads
        scale_patch_size = self.scale or 1./sqrt(E)
        score_patch_size = torch.einsum("blhe, bshe->bhls", queries_patch_size, key_patch_size) # bs*channel, nheads, patch_num, patch_num
        attn_patch_size = scale_patch_size*score_patch_size 
        series_patch_size = self.dp(torch.softmax(attn_patch_size, dim=-1)) # bs*channel, nheads, patch_num, patch_num

        #In-Patch
        B, L, H, E = queries_patch_num.shape # bs*channel, patch_size, nheads, d_model/nheads
        scale_patch_num = self.scale or 1./sqrt(E)
        score_patch_num = torch.einsum("blhe, bshe->bhls", queries_patch_num, key_patch_num) # bs*channel, nheads, patch_size, patch_size
        attn_patch_num = scale_patch_num*score_patch_num
        series_patch_num = self.dp(torch.softmax(attn_patch_num, dim=-1)) # bs*channel, nheads, patch_size, patch_size

        # Upsampling 
        series_patch_size = repeat(series_patch_size, "b l m n -> b l (m repeat_m) (n repeat_n)", repeat_m=self.patch_size[patch_index], repeat_n=self.patch_size[patch_index]) # bs*channels, nheads, patch_num*patch_size[patch_index], patch_num*patch_size[patch_index] 
        series_patch_num = series_patch_num.repeat(1, 1, self.window_size//self.patch_size[patch_index], self.window_size//self.patch_size[patch_index]) # bs*channels, nheads, patch_size*ws//patch_size[patch_index], patch_size*ws//patch_size[patch_index]
        series_patch_size = reduce(series_patch_size, "(b reduce_b) l m n -> b l m n", "mean", reduce_b=self.channel)
        series_patch_num = reduce(series_patch_num, "(b reduce_b) l m n -> b l m n", "mean", reduce_b=self.channel)

        return series_patch_size, series_patch_num
    

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, patch_size, channel, n_heads, win_size, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or d_model//n_heads
        d_values = d_values or d_model//n_heads
        self.inner_attention = attention
        self.patch_size = patch_size
        self.channel = channel
        self.windows_size = win_size
        self.n_heads = n_heads

        self.patch_query_proj = nn.Linear(d_model, d_keys*n_heads)
        self.patch_key_proj = nn.Linear(d_model, d_keys*n_heads)
        self.out_proj = nn.Linear(d_values*n_heads, d_model)
        self.value_proj = nn.Linear(d_model, d_values*n_heads)

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index):

        # patch_size
        B, L, M = x_patch_size.shape
        H = self.n_heads
        queries_patch_size, keys_patch_size = x_patch_size, x_patch_size
        queries_patch_size = self.patch_query_proj(queries_patch_size).view(B, L, H, -1)
        keys_patch_size = self.patch_key_proj(keys_patch_size).view(B, L, H, -1)


        # patch_num
        B, L, M = x_patch_num.shape
        H = self.n_heads
        queries_patch_num, keys_patch_num = x_patch_num, x_patch_num
        queries_patch_num = self.patch_query_proj(queries_patch_num).view(B, L, H, -1)
        keys_patch_num = self.patch_key_proj(keys_patch_num).view(B, L, H, -1)

        # x_ori
        B, L, _ = x_ori.shape
        values = self.value_proj(x_ori).view(B, L, H , -1)

        series, prior = self.inner_attention(queries_patch_size, queries_patch_num, keys_patch_size, keys_patch_num, patch_index)
        return series, prior
    
class Encoder(nn.Module):
    def __init__(self, attn_layers):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index):
        series_list = []
        prior_list = []

        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index)
            series_list.append(series)
            prior_list.append(prior)

        return series_list, prior_list
    
class DCDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        win_size = config.ws+1
        enc_in = config.in_dim
        n_heads = config.n_heads
        d_model = config.d_model
        e_layers = config.e_layers
        patch_size = config.patch_sizes
        dp = config.dp

        self.patch_size = patch_size
        self.enc_in = enc_in
        self.win_size = win_size

        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()

        for i, patchsize in enumerate(patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, d_model=d_model, dp=dp))
            self.embedding_patch_num.append(DataEmbedding(self.win_size//patchsize, d_model=d_model, dp=dp))

        self.embedding_window_size = DataEmbedding(enc_in, d_model, dp)

        self.encoder = Encoder(
            [
                AttentionLayer(
                    DAC_Structure(win_size=win_size, patch_size=patch_size, channel=enc_in),
                    d_model=d_model, patch_size=patch_size, channel=enc_in, n_heads=n_heads, win_size=win_size
                ) for l in range(e_layers)
            ]
        )

    def forward(self, x):
        B, L, M = x.shape #bs, ws, f
        series_patch_mean = []
        prior_patch_mean = []
        revin_layer = RevIN(num_features=M)
        x = revin_layer(x, "norm")
        x_ori = self.embedding_window_size(x) # bs, ws, d_model
        
        for patch_index, patchsize in enumerate(self.patch_size):

            x_patch_size, x_patch_num = x, x
            x_patch_size = rearrange(x_patch_size, "b l m -> b m l") # bs, f, ws
            x_patch_num = rearrange(x_patch_num, "b l m -> b m l") # bs, f, ws

            x_patch_size = rearrange(x_patch_size, "b m (n p) -> (b m) n p", p=patchsize) # bs*f, num_patches, patch_size
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size) # bs*f, num_patches, d_model

            x_patch_num = rearrange(x_patch_num, "b m (p n) -> (b m) p n", p=patchsize) # bs*f, patch_size, num_patches
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num) # bs*f, patch_size, d_model
            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            series_patch_mean.append(series), prior_patch_mean.append(prior)
            
        series_patch_mean = list(_flatten(series_patch_mean))
        prior_patch_mean = list(_flatten(prior_patch_mean))
        return series_patch_mean, prior_patch_mean
    

def my_kl_loss(p, q, eps=1e-4):
    res = p * (torch.log(p+eps) - torch.log(q+eps))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

class DCDetectorLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = DCDetector(config)
        self.lr = config.lr
        self.ws = config.ws + 1
        self.auc = BinaryAUROC()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        series, prior = self.model(x)

        series_loss = 0
        prior_loss = 0

        for i in range(len(prior)):
            prior_softmax = prior[i] / torch.unsqueeze(torch.sum(prior[i], dim=-1), dim=-1).repeat(1, 1, 1, self.ws) 

            series_loss += torch.mean(my_kl_loss(series[i], prior_softmax.detach())) + torch.mean(my_kl_loss(prior_softmax.detach(), series[i]))
            prior_loss += torch.mean(my_kl_loss(prior_softmax, series[i].detach())) + torch.mean(my_kl_loss(series[i].detach(), prior_softmax))

        series_loss=series_loss/len(prior)
        prior_loss=prior_loss/len(prior)

        loss = prior_loss - series_loss

        self.log('train_loss', loss)
        self.log('train_series_loss', series_loss)
        self.log('train_prior_loss', prior_loss)
    
        return loss

    def get_loss(self, x, mode=None):
        
        series, prior = self.model(x)

        loss=0

        for i in range(len(prior)):
            prior_softmax = prior[i] / torch.unsqueeze(torch.sum(prior[i], dim=-1), dim=-1).repeat(1, 1, 1, self.ws) # Softmax

            if i==0:
                loss = my_kl_loss(series[i], prior_softmax) + my_kl_loss(prior_softmax, series[i])
            else:
                loss += my_kl_loss(series[i], prior_softmax) + my_kl_loss(prior_softmax, series[i])

        metric = torch.softmax(loss, dim=-1)
        metric = metric[:, -1]
        errors = metric.detach().cpu()
        return errors

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def adjust_learning_rate(self, optimizer, epoch, lr_):
        lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        current_epoch = self.current_epoch
        initial_lr = self.lr
        self.adjust_learning_rate(optimizer, current_epoch, initial_lr)

    def test_step(self, batch, batch_idx):
        x, y = batch
        errors = self.get_loss(x, mode="test")
        self.auc.update(errors, y.int())
    
    def on_test_epoch_end(self):
        auc = self.auc.compute()
        self.auc.reset()
        self.log("auc", auc, prog_bar=True)