import torch 
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange
import math
import lightning as L 
from copy import deepcopy
from models.auc import StreamAUC


class Patcher(nn.Module):
    def __init__(self, window_size, patch_len):
        super().__init__()
        assert window_size % patch_len == 0, "window size must be divisible by patch length"
        self.window_size = window_size
        self.patch_len = patch_len
        self.patch_num = window_size// patch_len
        self.shape = {"window_size":self.window_size,
                              "patch_len":self.patch_len,
                              "patch_num":self.patch_num}

    def forward(self, window):

        # Input: 

        # x: bs x nvars x window_size

        # Output:

        # out: bs x nvars x patch_num x patch_len 
        patch_window = rearrange(window, 'b c (pn pl) -> b c pn pl', pl=self.patch_len)
        return patch_window
    
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

def PositionalEncoding(q_len, d_model):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return nn.Parameter(pe, requires_grad=False)

class _ScaledDotProduct(nn.Module):
    def __init__(self, d_model, n_heads, attn_dp=0.):
        super().__init__()

        self.attn_dp = nn.Dropout(attn_dp)
        head_dim = d_model//n_heads
        self.scale = head_dim**(-0.5)

    def forward(self, q, k, v, prev=None):
        
        # Input: 

        # q: bs x nheads x num_patches x d_k
        # k: bs x nheads x d_k x num_patches
        # v: bs x nheads x num_patches x d_v
        # prev: bs x nheads x num_patches x num_patches

        # Output:

        # out: bs x nheads x num_patches x d_v
        # attn_weights: bs x nheads x num_patches x num_patches
        # attn_scores: bs x nheads x num_patches x num_patches

        attn_scores = torch.matmul(q, k)*self.scale

        if prev is not None: attn_scores+=prev

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dp(attn_weights)

        out = torch.matmul(attn_weights, v)
        
        return out, attn_scores
    

class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, attn_dp=0., proj_dp=0., qkv_bias=True):
        super().__init__()
        d_k = d_model//n_heads if d_k is None else d_k
        d_v = d_model//n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, n_heads*d_k, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, n_heads*d_k, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, n_heads*d_v, bias=qkv_bias)

        self.sdp = _ScaledDotProduct(d_model=d_model, n_heads=n_heads, attn_dp=attn_dp)

        self.to_out = nn.Sequential(nn.Linear(n_heads*d_v, d_model), nn.Dropout(proj_dp))

    def forward(self, Q, K=None, V=None, prev=None):

        # Input: 

        # Q: bs x num_patches x d_model
        # K: bs x num_patches x d_model
        # V: bs x num_patches x d_model
        # prev: bs x num_patches x num_patches

        # Output:

        # out: bs x num_patches x d_model
        # attn_scores: bs x num_patches x num_patches

        bs = Q.size(0)
        if K is None: K = Q.clone()
        if V is None: V = Q.clone()
        
        q = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)
        k = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)

        out, attn_scores = self.sdp(q, k, v, prev=prev)

        out = out.transpose(1, 2).contiguous().view(bs, -1, self.n_heads*self.d_v)
        out = self.to_out(out)

        return out, attn_scores
    
class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, attn_dp=0., dp=0.):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.self_attn = _MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, attn_dp=attn_dp, proj_dp=dp)
        self.attn_dp = nn.Dropout(attn_dp)
        self.norm_attn = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Dropout(dp),
                                nn.Linear(d_ff, d_model))
        
        self.ffn_dp = nn.Dropout(dp)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, src, prev):

        # Input: 

        # src: bs x num_patches x d_model
        # prev: bs x n_heads x num_patches x num_patches

        # Output:

        # out: bs x num_patches x d_model
        # attn_scores: bs x nheads x num_patches x num_patches

        src, scores = self.self_attn(Q=src, prev=prev)
        src = self.attn_dp(src)
        src = self.norm_attn(src)

        src2 = self.ff(src)

        src = src + self.ffn_dp(src2)
        src = self.norm_ffn(src)

        return src, scores
    

class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, attn_dp=0., dp=0., n_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([TSTEncoderLayer(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, 
                                                     d_ff=d_ff, attn_dp=attn_dp, dp=dp) for _ in range(n_layers)])
        
    def forward(self, x):

        # Input: 

        # x: bs x num_patches x d_model

        # Output:

        # out: bs x num_patches x d_model
        out=x
        prev=None
        for layer in self.layers:
            out, prev = layer(out, prev=prev)
        return out
    

class TSTiEncoder(nn.Module):
    def __init__(self, patch_num, patch_len, d_model, n_heads, n_layers=3, d_ff=256, attn_dp=0., dp=0.):
        super().__init__()
        self.patch_num, self.patch_len = patch_num, patch_len

        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = PositionalEncoding(q_len=patch_num, d_model=d_model)
        self.dp=nn.Dropout(dp)
        
        self.encoder = TSTEncoder(d_model=d_model, n_heads=n_heads, d_ff=d_ff, attn_dp=attn_dp, dp=dp, n_layers=n_layers)

    def forward(self, x):

        # Input: 

        # x: bs x nvars x num_patches  x patch_len

        # Output:

        # out: bs x nvars x d_model x num_patches

        n_vars = x.shape[1]
        x = self.W_P(x) # bs x nvars x num_patches x d_model

        x = torch.reshape(x, (x.shape[0]*x.shape [1], x.shape[2], x.shape[3])) # bs*nvars x num_patches x d_model    (channel indep)
        x = self.dp(x+self.W_pos)
        x = self.encoder(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1])) # bs x nvars x num_patches x d_model

        return x  # bs x nvars x num_patches x d_model
    

class PatchTrADencoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        window_size = config.ws
        patch_len = config.patch_len
        d_model = config.d_model
        n_heads = config.n_heads
        n_layers = config.n_layers
        d_ff = config.d_ff
        attn_dp=0.
        dp=0.3

        self.patcher = Patcher(window_size=window_size, patch_len=patch_len)
        shape = self.patcher.shape
        patch_num = shape["patch_num"]

        self.encoder = TSTiEncoder(patch_num=patch_num, patch_len=patch_len, d_model=d_model, 
                                   n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, attn_dp=attn_dp,
                                   dp=dp)
        self.tp = Transpose(1, 2)
    

    def forward(self, x):
        # Input: 

        # x: bs x window_size x nvars
        
        patched = self._get_patch(x) # bs x nvars x patch_len x patch_num
        
        h = self.encoder(patched) # bs x nvars x patch_num x d_model

        return patched, h
    
    def _get_patch(self, x):
        x = self.tp(x) # bs x nvars x window_size
        patched = self.patcher(x) # bs x nvars x patch_num x patch_len
        return patched
    
class Head(nn.Module):
    def __init__(self, n_vars, patch_len, patch_num,  d_model, head_dp=0.):
        super().__init__() 

        self.n_vars = n_vars

        self.layers = nn.ModuleList([])
        for _ in range(n_vars):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(d_model, patch_len),
                    nn.Dropout(head_dp),
                )
            )

    def forward(self, x):

        # Input: 

        # x: bs x nvars x num_patches x d_model

        # Output:

        # out: bs x nvars x num_patches x patch_len

        outs = []
        for i in range(self.n_vars):
            input = x[:, i, :, :]
            out = self.layers[i](input)
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        return outs
    
class PatchTrAD(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        window_size = 100
        num_patches = window_size // config.patch_len
        config["ws"] = window_size

        self.encoder = PatchTrADencoder(config)
        checkpoint_path = config.save_path +'.ckpt'
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        self.encoder.load_state_dict(checkpoint)
        self.encoder.requires_grad_(False if config.freeze_encoder else True)

        self.head = Head(config.in_dim, config.patch_len, num_patches, config.d_model, config.head_dp if config.head_dp else 0)
        self.head.requires_grad_(True)

    def forward(self, x):
        patched, h = self.encoder(x)

        out = self.head(h)
        return patched, out
    
    def get_loss(self, x, mode="train"):

        inp, out = self.forward(x)

        if mode=="train":
            error = ((out - inp)**2).flatten(start_dim=1).mean(dim=(1))
            
        elif mode=="test":
            inp = inp[:, :, -1, :]
            out = out[:, :, -1, :]
            error = ((out - inp)**2).flatten(start_dim=1).mean(dim=(1))

        return error
    
class JEPAtchTradLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = PatchTrAD(config)
        self.lr = config.lr
        self.auc = StreamAUC()
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.model.get_loss(x, mode="train")
        loss = loss.mean()
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def get_loss(self, x, mode=None):
        return self.model.get_loss(x, mode=mode)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        errors = self.get_loss(x, mode="test")
        self.auc.update(errors, y.int())
    
    def on_test_epoch_end(self):
        auc = self.auc.compute()
        self.auc.reset()
        self.log("auc", auc, prog_bar=True)