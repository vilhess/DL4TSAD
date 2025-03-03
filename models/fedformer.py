# Modified Code of the original repository focus on the Fourier version; 
# corrected some things:
# - works with different head size
# - correction fourier cross attention
# - works for data without categorical features known

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import math
import lightning as L 

class TokenEmbedding(nn.Module):
    def __init__(self,c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
    
    def forward(self, x):
        x = self.tokenConv(x.transpose(1, 2)).transpose(1, 2)
        return x
    
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, cat_size, d_model):
        super().__init__()
        self.emb = nn.Linear(cat_size, d_model, bias=False)
    def forward(self, x):
        return self.emb(x)
    
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, cat_size, d_model, dp=0.1):
        super().__init__()
        self.cat_size = cat_size
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        if cat_size is not None:
            self.temporal_embedding = TimeFeatureEmbedding(cat_size=cat_size, d_model=d_model)
        self.dp = nn.Dropout(dp)
    def forward(self, x, x_mark=None):
        x = self.value_embedding(x) 
        if self.cat_size is not None:
            x = x + self.temporal_embedding(x_mark)
        return x
    
def get_frequency_modes(seq_len, modes):
    modes = min(modes, seq_len//2)
    index = list(range(0, seq_len//2))
    np.random.shuffle(index)
    index = index[:modes]
    index.sort()
    return index

class myLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)
    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias

class moving_avg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        
    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, self.kernel_size-1 - math.floor((self.kernel_size - 1)//2) , 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1)//2) , 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
    
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size=kernel_size)
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class series_decomp_multi(nn.Module):
    def __init__(self, kernel_sizes):
        super().__init__()
        self.moving_avg = [moving_avg(kernel_size=kernel) for kernel in kernel_sizes]
        self.layer = nn.Linear(1, len(kernel_sizes))
    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, n_heads, modes):
        super().__init__()
        self.index = get_frequency_modes(seq_len, modes)
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(n_heads, in_channels // n_heads, out_channels // n_heads, len(self.index), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)
    
    def forward(self, q, k, v):
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
    
class FourierCrossAttention(nn.Module):
    def __init__(self, in_c, out_c, seq_len_q, seq_len_kv, n_heads, modes):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.index_q = get_frequency_modes(seq_len_q, modes=modes)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes)
        self.scale = 1/ (in_c*out_c)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(n_heads, in_c // n_heads, out_c // n_heads, len(self.index_q), dtype=torch.cfloat))
        
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)
    
    def forward(self, q, k, v):
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xk.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        xv_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xv.device, dtype=torch.cfloat)
        xv_ft = torch.fft.rfft(xv, dim=-1)
        for i, j in enumerate(self.index_kv):
            xv_ft_[:, :, :, i] = xv_ft[:, :, :, j]

        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)
        xqk_ft = xqk_ft.tanh()
        xqkv_ft = torch.einsum('bhxy,bhey->bhex', xqk_ft, xv_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L//2 +1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:,:,:,j] = xqkvw[:,:,:,i]
        out = torch.fft.irfft(out_ft /self.in_c / self.out_c, n=xq.size(-1))
        return out
    
class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_correlation(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)
        return self.out_projection(out)
    
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, moving_avg, dp):
        super().__init__()
        d_ff = d_ff 
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(kernel_sizes=moving_avg)
            self.decomp2 = series_decomp_multi(kernel_sizes=moving_avg)
        else:
            self.decomp1 = series_decomp(kernel_size=moving_avg)
            self.decomp2 = series_decomp(kernel_size=moving_avg)

        self.dp = nn.Dropout(dp)
        self.activation = nn.GELU()

    def forward(self, x):
        new_x = self.attention(x, x, x)
        x = x + self.dp(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dp(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dp(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(y+x)
        return res
    
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layers):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layers

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)   
        x = self.norm(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, self_attn, cross_attn, d_model, c_out, d_ff, moving_avg, dp):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(kernel_sizes=moving_avg)
            self.decomp2 = series_decomp_multi(kernel_sizes=moving_avg)
            self.decomp3 = series_decomp_multi(kernel_sizes=moving_avg)
        else:
            self.decomp1 = series_decomp(kernel_size=moving_avg)
            self.decomp2 = series_decomp(kernel_size=moving_avg)
            self.decomp3 = series_decomp(kernel_size=moving_avg)
        
        self.dp = nn.Dropout(dp)
        self.proj = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1, padding_mode="circular", bias=False)
        self.activation = nn.GELU()

    def forward(self, x, cross):
        x = x + self.dp(self.self_attn(x, x, x))
        x, trend1 = self.decomp1(x)
        x = x + self.dp(self.cross_attn(x, cross, cross))
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dp(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dp(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x+y)
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.proj(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend
    
class Decoder(nn.Module):
    def __init__(self, layers, norm_layer, projection):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.proj = projection

    def forward(self, x, cross, trend):
        for layer in self.layers:
            x, residual_trend = layer(x, cross)
            trend = trend + residual_trend
        x = self.norm(x)
        x = self.proj(x)
        return x, trend

class FEDformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        seq_len = config.ws
        label_len = seq_len//2
        pred_len = 1
        modes = config.modes
        enc_in = config.in_dim
        dec_in = config.in_dim
        c_out = config.in_dim
        cat_size = None
        moving_avg = list(config.moving_avg)
        e_layers = config.e_layers
        d_layers = config.d_layers
        n_heads = config.n_heads
        d_model = config.d_model
        d_ff = config.d_ff
        dp = config.dp

        self.pred_len = pred_len
        self.label_len = label_len
        self.cat_size = cat_size

        if isinstance(moving_avg, list):
            self.decomp = series_decomp_multi(moving_avg)
        else:
            self.decomp = series_decomp(moving_avg)

        self.enc_embedding = DataEmbedding_wo_pos(cat_size=cat_size, c_in=enc_in, d_model=d_model, dp=dp)
        self.dec_embedding = DataEmbedding_wo_pos(cat_size=cat_size, c_in=dec_in, d_model=d_model, dp=dp)

        encoder_self_attn = FourierBlock(in_channels=d_model, out_channels=d_model, seq_len=seq_len, n_heads=n_heads, modes=modes)
        decoder_self_attn = FourierBlock(in_channels=d_model, out_channels=d_model, seq_len=label_len + pred_len, n_heads=n_heads, modes=modes)
        decoder_cross_attn = FourierCrossAttention(in_c=d_model, out_c=d_model, seq_len_q=label_len + pred_len, seq_len_kv=seq_len, n_heads=n_heads, modes=modes)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_attn, d_model=d_model, n_heads=n_heads),
                    d_model, d_ff, moving_avg, dp
                ) for l in range(e_layers)
            ],
            norm_layers=myLayerNorm(d_model)
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_attn, d_model=d_model, n_heads=n_heads),
                    AutoCorrelationLayer(decoder_cross_attn, d_model=d_model, n_heads=n_heads),
                    d_model, c_out, d_ff, moving_avg, dp
                ) for l in range(d_layers)
            ],
            norm_layer=myLayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )
    
    def forward(self, x_enc, x_enc_mark=None, x_dec_mark=None):
        # x_enc: B,seq_len,F ; x_enc_mark: B,seq_len,F_mark ; x_dec_mark: B,label+pred len,F_mark ; 

        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1) # B,1,F -> B, pred_len, F
        seasonal_init, trend_init = self.decomp(x_enc) # B, seq_len, F
        trend_init = torch.cat([trend_init[:,-self.label_len:,:], mean], dim=1) # B, label+pred len, F
        seasonal_init = F.pad(seasonal_init[:,-self.label_len:, :], (0, 0, 0, self.pred_len)) # # B, label+pred len, F (rajoute des 0 de label_len jusqua pred_len)
        enc_in = self.enc_embedding(x_enc, x_enc_mark) # B, seq_len, D
        enc_out = self.encoder(enc_in) # B, seq_len, D
        dec_in = self.dec_embedding(seasonal_init, x_dec_mark) # B, label+pred len, D
        seasonal_part, trend_part = self.decoder(dec_in, enc_out, trend=trend_init)
        decoder_out = trend_part + seasonal_part
        return decoder_out[:,-self.pred_len:, :]

class FEDformerLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = FEDformer(config)
        self.lr = config.lr
        self.criterion = nn.MSELoss()
        self.criterion_wise = nn.MSELoss(reduction="none")

    def training_step(self, batch, batch_idx):
        x, _ = batch
        inputs = x[:,:-1,:]
        target = x[:,-1,:]
        pred = self.model(inputs)
        loss = self.criterion(pred.squeeze(1), target)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def get_loss(self, x, mode=None):
        inputs = x[:,:-1,:]
        target = x[:,-1,:]
        pred = self.model(inputs)
        loss = self.criterion_wise(target, pred.squeeze(1))
        return loss