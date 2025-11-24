import torch 
import torch.nn as nn 
import numpy as np 
from math import sqrt, pi, log
import lightning as L 
from models.scorer import StreamScorer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad=False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * (- log(10000.)/d_model)).exp()

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
    
    def forward(self, x): # b, w, feat
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dp=0.):
        super().__init__()
        self.value_embed = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.pos_embed = PositionalEncoding(d_model=d_model)
        self.dp = nn.Dropout(dp)

    def forward(self, x):
        x = self.value_embed(x) 
        x = x + self.pos_embed(x)
        return self.dp(x)
    
class AnomalyAttention(nn.Module):
    def __init__(self, win_size, scale=None, attention_dp=0.):
        super().__init__()
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device="cuda"
        self.scale=scale
        self.dp = nn.Dropout(attention_dp)
        self.distances = torch.zeros((win_size, win_size), device=self.device)
        
        for i in range(win_size):
            for j in range(win_size):
                self.distances[i, j] = abs(i-j)

    def forward(self, queries, keys, values, sigma):
        B, L, H, E = queries.shape # B, L, H, Dk
        _, S, _, D = values.shape # B, S, H, Dv
        scale = self.scale or 1./sqrt(E)
        
        scores = torch.einsum("blhe, bshe->bhls", queries, keys) # B, H, L, S
        attn = scale*scores # B, H, L, S
        series = self.dp(torch.softmax(attn, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", series, values) # B, W, H, Dv

        sigma = sigma.transpose(1, 2) # B, H, L
        ws = sigma.shape[-1]
        sigma = torch.sigmoid(sigma*5)+1e-5
        sigma = torch.pow(3, sigma) -1 # B, H, L
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, ws) # B, H, L, L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1) # B, H, L, L
        prior = 1./(sqrt(2*pi)*sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        return (V.contiguous(), series, prior, sigma)
    
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()

        d_keys = d_keys or d_model//n_heads
        d_values = d_values or d_model//n_heads
        self.norm = nn.LayerNorm(d_model)
        self.inner_attn = attention
        self.query_proj = nn.Linear(d_model, d_keys*n_heads)
        self.key_proj = nn.Linear(d_model, d_keys*n_heads)
        self.value_proj = nn.Linear(d_model, d_values*n_heads)
        self.sigma_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads*d_values, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape # B, W, Dmodel
        _, S, _ = keys.shape # B, W, Dmodel
        H = self.n_heads
        x = queries

        queries = self.query_proj(queries).reshape(B, L, H, -1) # B, L, H, Dk
        keys = self.key_proj(keys).reshape(B, S, H, -1) # B, S, H, Dk
        values = self.value_proj(values).reshape(B, S, H, -1) # B, S, H, Dv
        sigma = self.sigma_proj(x).reshape(B, L, H) # B, L, H

        out, series, prior, sigma = self.inner_attn(queries, keys, values, sigma)
        out = out.view(B, L, -1)
        return self.out_proj(out), series, prior, sigma
    
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dp=0.1):
        super().__init__()
        d_ff = d_ff or d_model*4
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dp = nn.Dropout(dp)
        self.activation = nn.ReLU()

    def forward(self, x):
        new_x, attn, mask, sigma = self.attention(x, x, x)
        x = x + self.dp(new_x)
        y = x = self.norm1(x)
        y = self.dp(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dp(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma
    
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layers=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layers

    def forward(self, x):
        # x : b, w, d
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        if self.norm is not None:
            x = self.norm(x)
        return x, series_list, prior_list, sigma_list
    
class AnomalyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        enc_in = config.in_dim
        d_model = config.d_model
        dp = config.dp
        win_size = config.ws+1
        n_heads = config.n_heads
        d_ff = config.d_ff
        e_layers = config.e_layers

        self.embedding = DataEmbedding(c_in=enc_in, d_model=d_model, dp=dp)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(attention=AnomalyAttention(win_size=win_size, attention_dp=dp), d_model=d_model, n_heads=n_heads),
                    d_model=d_model, d_ff=d_ff, dp=dp
                ) for l in range(e_layers)
            ], norm_layers=nn.LayerNorm(d_model)
        )
        self.proj = nn.Linear(d_model, enc_in, bias=True)

    def forward(self, x):
        # x : b, w, f
        enc_out = self.embedding(x) # b, w, d_model
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.proj(enc_out)
        return enc_out, series, prior, sigmas
    
def my_kl_loss(p, q, eps=1e-4):
    res = p * (torch.log(p+eps) - torch.log(q+eps))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

class AnomalyTransformerLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = AnomalyTransformer(config)
        self.ws = config.ws+1
        self.lr = config.lr
        self.K = config.K
        self.criterion = nn.MSELoss()
        self.criterion_wise = nn.MSELoss(reduction="none")

        self.automatic_optimization = False
        self.scorer = StreamScorer(config.metrics)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        opt = self.optimizers()
        opt.zero_grad()
        out, series, prior, _ = self.model(x)

        series_loss = 0
        prior_loss = 0

        for u in range(len(prior)):
            s = series[u]
            p = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.ws)
            series_loss+=torch.mean(my_kl_loss(s, p.detach())) + torch.mean(my_kl_loss(p.detach(), s))
            prior_loss+=torch.mean(my_kl_loss(s.detach(), p)) + torch.mean(my_kl_loss(p, s.detach()))

        series_loss = series_loss/len(prior)
        prior_loss = prior_loss/len(prior)
        rec_loss = self.criterion(x, out)

        loss1 = rec_loss - self.K*series_loss
        loss2 = rec_loss + self.K*prior_loss

        self.manual_backward(loss1, retain_graph=True)
        self.manual_backward(loss2, retain_graph=True)
        
        self.log("rec_loss", rec_loss)
        self.log("series_loss", series_loss)
        self.log("prior_loss", prior_loss)
        self.log("loss1", loss1)
        self.log("loss2", loss2)    

        opt.step()


    def get_loss(self, x, mode=None):
        out, series, prior, sigmas = self.model(x)

        series_loss = 0
        prior_loss = 0

        loss = torch.mean(self.criterion_wise(x, out), dim=-1)

        for u in range(len(prior)):
            s = series[u]
            p = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.ws)

            if u==0:
                series_loss = my_kl_loss(s, p.detach())
                prior_loss = my_kl_loss(p, s.detach())
            else:
                series_loss += my_kl_loss(s, p.detach())
                prior_loss += my_kl_loss(p, s.detach())


        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        metric = metric * loss
        metric = metric[:, -1]
        loss = metric.detach().cpu()
        return loss

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
        self.scorer.update(errors, y.int())
    
    def on_test_epoch_end(self):
        metrics = self.scorer.compute()
        self.scorer.reset()
        for k, v in metrics.items():
            self.log(f"test_{k}", v, prog_bar=True)