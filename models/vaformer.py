import torch
import torch.nn as nn
import math 
from einops import rearrange
import lightning as L
import torch.optim as optim
from rotary_embedding_torch import RotaryEmbedding

from sklearn.metrics import roc_auc_score

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
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

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
    
def PositionalEncoding(q_len, d_model, normalize=True, learn_pe=False):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return nn.Parameter(pe, requires_grad=learn_pe)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, rope_kind=None):
        super().__init__()
        assert d_model%n_heads==0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.head_dim = d_model//n_heads
        self.n_heads = n_heads

        self.rope_kind = rope_kind
        if self.rope_kind:
            self.rope = RotaryEmbedding(dim=self.head_dim//2)
    
    def forward(self, q, k=None, v=None):
        bs, context, dim = q.size()

        if k is None:
            k = q
        if v is None:
            v = q

        q = self.WQ(q).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.WQ(k).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.WQ(v).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rope_kind=="q":
            q = self.rope.rotate_queries_or_keys(q)

        elif self.rope_kind=="qk":
            q  = self.rope.rotate_queries_or_keys(q)
            k = self.rope.rotate_queries_or_keys(k)

        attn_weights = q @ k.transpose(2, 3) * self.head_dim**(-1/2)
        scores = torch.nn.functional.softmax(attn_weights, dim=-1)
        scores = self.dropout(scores)

        values = scores @ v

        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.layers(x)
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, rope_kind=None)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, dropout=dropout)
    
    def forward(self, x):
        out_attn = self.attn(self.ln1((x)))
        x = x + out_attn
        out = x + self.ff(self.ln2(x))
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, rope_kind="qk")
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout,  rope_kind="q")
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, dropout=dropout)
    
    def forward(self, x, enc_out):
        out_attn = self.self_attn(q=self.norm1(x))
        x = x + out_attn

        out_cross = self.cross_attn(q=self.norm2(x), k=enc_out, v=enc_out)
        x = x + out_cross

        x = x + self.ff(self.norm3(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_pre_layers, n_layers, dropout=0.1):
        super().__init__()
        self.pre_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_pre_layers)
            ]
        )
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, enc_out):
        for layer in self.pre_layers:
            x = layer(x)
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)


class VAformer(nn.Module): # encoder tokens=variables decoder tokens=signal patches
    def __init__(self, in_dim, seq_len, patch_len, target_len, d_model, 
                 n_heads, n_layers_encoder, n_prelayers_decoder, 
                 n_layers_decoder, revin=True, dropout=0.1):
        super().__init__()
        assert seq_len%patch_len==0, f"seq_len ({seq_len}) should be divisible by patch_len ({patch_len})"

        self.revin = revin
        if self.revin:
            self.revin = RevIN(num_features=in_dim, affine=True)

        # encoder part
        self.proj1 = nn.Linear(seq_len, d_model)
        self.embedding = nn.Parameter(torch.randn(1, in_dim, d_model), requires_grad=True)
        self.dp = nn.Dropout(dropout)
        self.transformer_encoder = TransformerEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers_encoder ,dropout=dropout)

        # decoder part
        self.patch_len = patch_len
        self.patch_num = seq_len//patch_len
        self.in_dim = in_dim
        self.proj2 = nn.Linear(patch_len, d_model)
        self.W_pos = PositionalEncoding(q_len=seq_len//patch_len, d_model=d_model)
        self.decoder = TransformerDecoder(d_model=d_model, n_heads=n_heads, n_pre_layers=n_prelayers_decoder, n_layers=n_layers_decoder, dropout=dropout)

        self.forecaster = nn.ModuleList(
            [
                nn.Linear(d_model*self.patch_num, target_len)
                for _ in range(in_dim)
            ]
        )
    
    def forward(self, x): 
        bs, ws, in_dim = x.size()

        if self.revin:
            x = self.revin(x, mode="norm")

        # encoder part
        x_enc = x.transpose(1, 2)
        x_enc = self.proj1(x_enc) # bs, in_dim, d_model
        x_enc = self.dp(x_enc + self.embedding)
        x_enc = self.transformer_encoder(x_enc) # bs, in_dim, d_model

        # decoder part
        x_dec = x.transpose(1, 2)
        x_dec = x_dec.reshape(-1, ws)
        x_dec = rearrange(x_dec, "b (pn pl) -> b pn pl", pl=self.patch_len)

        x_dec = self.proj2(x_dec)
        #x_dec = self.dp(x_dec + self.W_pos)
        x_dec = self.dp(x_dec) ##

        x_enc = x_enc.repeat_interleave(repeats=in_dim, dim=0)
        
        x_dec = self.decoder(x_dec, x_enc)

        # forecasting part
        x_dec = x_dec.reshape(bs, self.in_dim, self.patch_num, -1)
        x_dec = x_dec.flatten(start_dim=2)

        forecasting = []
        for i, proj in enumerate(self.forecaster):
            forecasting.append(proj(x_dec[:, i, :]))
        forecasting = torch.stack(forecasting).permute(1, 2, 0)
        
        if self.revin:
            forecasting = self.revin(forecasting, mode="denorm")

        return forecasting

    
class StreamAUC:
    def __init__(self):
        self.test_scores = []
        self.test_labels = []
    
    def update(self, errors, labels):

        self.test_scores.append(errors)
        self.test_labels.append(labels)
    
    def compute(self):
        self.test_scores = torch.cat(self.test_scores).detach().cpu().numpy()
        self.test_labels = torch.cat(self.test_labels).detach().cpu().numpy()

        auc = roc_auc_score(y_true=self.test_labels, y_score=self.test_scores)
        return auc
    
    def reset(self):
        self.test_scores = []
        self.test_labels = []
    
class VAformerLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = VAformer(in_dim=config.in_dim,
                              seq_len=config.ws,
                              patch_len=config.patch_len,
                              target_len=1,
                              d_model=config.d_model,
                              n_heads=config.n_heads,
                              n_layers_encoder=config.n_layers_encoder,
                              n_prelayers_decoder=config.n_prelayers_decoder,
                              n_layers_decoder=config.n_layers_decoder,
                              revin=config.revin,
                              dropout=config.dropout)
        
        self.auc = StreamAUC()

        self.save_hyperparameters(config)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.get_loss(x, mode="train")
        loss = loss.mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def get_loss(self, x, mode=None):
        ctx, tar = x[:, :-1, :], x[:, -1, :]
        pred = self.model(ctx)
        loss = torch.sum((pred.squeeze(1) - tar)**2, dim=1)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        errors = self.get_loss(x, mode="test")

        self.auc.update(errors, y)
    
    def on_test_epoch_end(self):
        auc = self.auc.compute()
        self.log("auc", auc, prog_bar=True)
        self.auc.reset()