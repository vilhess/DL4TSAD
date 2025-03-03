import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import lightning as L 
import numpy as np
import math

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, q_len=500):
        super().__init__()

        pe = torch.zeros(q_len, d_model)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return  x + self.pe[:x.size(1), :]

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
    
class DataEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.token_embed = TokenEmbedding(c_in=in_dim, d_model=out_dim)
        self.pe = PositionalEncoding(d_model=out_dim)

    def forward(self, x):
        return self.pe(self.token_embed(x))
    
def get_frequency_modes(seq_len, modes):
    modes = min(modes, seq_len//2)
    index = list(range(0, seq_len//2))
    np.random.shuffle(index)
    index = index[:modes]
    index.sort()
    return index

class FourierSelfAtt(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, modes):
        super().__init__()
        self.scale = (d_model // n_heads)**-.5
        self.indices_q = get_frequency_modes(seq_len, modes)
        self.indices_k = get_frequency_modes(seq_len, modes)
        self.indices_v = get_frequency_modes(seq_len, modes)

    def forward(self, q, k, v):
        B, H, L, D = q.shape
        q_ft = torch.fft.rfft(q, dim=2)
        k_ft = torch.fft.rfft(k, dim=2)
        v_ft = torch.fft.rfft(v, dim=2)

        q_ft_ = torch.zeros(B, H, len(self.indices_q), D, device=q_ft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.indices_q):
            q_ft_[:, :, i, :] = q_ft[:, :, j, :]

        k_ft_ = torch.zeros(B, H, len(self.indices_k), D, device=k_ft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.indices_k):
            k_ft_[:, :, i, :] = k_ft[:, :, j, :]

        v_ft_ = torch.zeros(B, H, len(self.indices_v), D, device=v_ft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.indices_v):
            v_ft_[:, :, i, :] = v_ft[:, :, j, :]
        qk_ft = torch.matmul(q_ft_, k_ft_.transpose(2, 3))*self.scale
        scores = qk_ft.tanh()
        values = torch.matmul(scores, v_ft_)
        out = torch.fft.irfft(values, dim=2, n=L)

        return out
    
class SelfAtt(nn.Module):
    def __init__(self, d_model, n_heads, mask_prob):
        super().__init__()
        self.scale = (d_model // n_heads)**-.5
        self.prob = mask_prob

    def forward(self, q, k, v):
        qk_ft = torch.matmul(q, k.transpose(2, 3)) * self.scale
        scores = qk_ft.tanh()

        if self.training:
            mask = torch.bernoulli(torch.full((q.size(0), q.size(1), q.size(2), q.size(2)), self.prob)).bool().to(q.device)
            scores = scores * (~mask).float()  # Use multiplication instead of in-place fill

        values = torch.matmul(scores, v)
        return values

    
class AutoCorrelationLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        subdim = d_model//n_heads
        self.n_heads = n_heads

        self.query_proj = nn.Linear(d_model, subdim*n_heads)
        self.key_proj = nn.Linear(d_model, subdim*n_heads)
        self.value_proj = nn.Linear(d_model, subdim*n_heads)
        self.attention = attention
        self.out_proj = nn.Linear(subdim*n_heads, d_model)

    def forward(self, x):
        Q = self.query_proj(x).reshape(x.size(0), self.n_heads, x.size(1), -1)
        K = self.key_proj(x).reshape(x.size(0), self.n_heads, x.size(1), -1)
        V = self.value_proj(x).reshape(x.size(0), self.n_heads, x.size(1), -1)

        out = self.attention(Q, K, V)
        out = out.transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        
        out = self.out_proj(out)
        return out
    
class Encoder(nn.Module):
    def __init__(self, correlation, d_model, d_ff, attn_dp, ffn_dp):
        super().__init__()
        self.correlation = correlation
        self.dp = nn.Dropout(attn_dp)
        self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model))
        self.ff = nn.Sequential(nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False),
                                nn.GELU(),
                                nn.Dropout(ffn_dp),
                                nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False))
        self.ff_dp = nn.Dropout(ffn_dp)
        self.norm_ffn = nn.Sequential(nn.BatchNorm1d(d_model), Transpose(1, 2))

    def forward(self, x):
        x = self.correlation(x)
        x = self.dp(x)
        x = self.norm_attn(x)
        x2 = self.ff(x)
        x = x + self.ff_dp(x2)
        x = self.norm_ffn(x)

        return x
    
class FDAD(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_dim = config.in_dim
        seq_len = config.ws+1
        d_model = config.d_model
        d_ff = config.d_ff
        modes = config.modes
        n_heads = config.n_heads
        attn_dp = config.attn_dp
        ffn_dp = config.ffn_dp 
        mask_prob = config.mask_prob

        self.embedder = DataEmbedding(in_dim=in_dim, out_dim=d_model)

        mhsa = SelfAtt(d_model=d_model, n_heads=n_heads, mask_prob=mask_prob)
        fmhsa = FourierSelfAtt(d_model=d_model, n_heads=n_heads, seq_len=seq_len, modes=modes)

        corr = AutoCorrelationLayer(attention=mhsa, d_model=d_model, n_heads=n_heads)
        fcorr = AutoCorrelationLayer(attention=fmhsa, d_model=d_model, n_heads=n_heads)

        self.encoder = Encoder(correlation=corr, d_model=d_model, d_ff=d_ff, attn_dp=attn_dp, ffn_dp=ffn_dp)
        self.fencoder = Encoder(correlation=fcorr, d_model=d_model, d_ff=d_ff, attn_dp=attn_dp, ffn_dp=ffn_dp)

        self.final_proj = nn.Linear(d_model, in_dim)

    def forward(self, x):
        x = self.embedder(x)
        x1 = self.encoder(x)
        x2 = self.fencoder(x)
        out = x1 + x2
        out = self.final_proj(out)
        return out
    
class FDADLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = FDAD(config)
        self.lr = config.lr
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed = self.model(x)
        loss = F.mse_loss(x, reconstructed, reduction="mean")
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_loss(self, x, mode=None):
        reconstructed = self.model(x)
        return ((x - reconstructed)**2).mean(dim=(1, 2))