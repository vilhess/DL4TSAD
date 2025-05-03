import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math
import lightning as L
from torchmetrics.classification import BinaryAUROC


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, seq_len=1000):
        super().__init__()
        pe = torch.zeros(seq_len, dim)
        position = torch.linspace(0, seq_len, seq_len).unsqueeze(1)
        value = torch.exp( -math.log(10000) * torch.arange(0, dim, 2) / dim).unsqueeze(0)
        pe[:, ::2] = torch.sin(position*value)
        pe[:, 1::2] = torch.cos(position*value)
        self.register_buffer('pe', pe)

    def forward(self, time):
        return self.pe[time, :]
    
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_dim)
        self.time_proj = nn.Linear(time_dim, in_dim)
        self.act = nn.ReLU()
        
    def forward(self, x, t):
        t = self.time_proj(t).unsqueeze(-1)
        x = x + t
        x = self.act(self.bn(self.conv(x)))
        return x
    
class DownBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class UpBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class FastUnet(nn.Module):
    def __init__(self, dims, time_dim):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )

        self.downs = nn.ModuleList([])
        for i in range(len(dims)-1):
                self.downs.append(nn.ModuleList([ConvBlock(dims[i], dims[i+1], time_dim=time_dim), DownBlock(dims[i+1])]))

        mid_dim = dims[-1]
        self.mids = nn.ModuleList([
            ConvBlock(mid_dim, 2*mid_dim, time_dim=time_dim),
            ConvBlock(2*mid_dim, 2*mid_dim, time_dim=time_dim),
            ConvBlock(2*mid_dim, mid_dim, time_dim=time_dim)
        ])

        dims = list(reversed(dims))
        self.ups = nn.ModuleList([])
        for i in range(len(dims)-1):
                self.ups.append(nn.ModuleList([UpBlock(dims[i]), ConvBlock(dims[i], dims[i+1], time_dim=time_dim)]))
        
        self.final_conv = nn.Conv1d(in_channels=dims[-1], out_channels=dims[-1], kernel_size=3, padding=1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        
        h = []
        for conv, down in self.downs:
            x = conv(x,t)
            h.append(x)
            x = down(x)
        
        for mid in self.mids:
            x = mid(x, t)
        
        for up, conv in self.ups:
            prev = h.pop()
            x = up(x)
            x = x + prev
            x = conv(x, t)
        x = self.final_conv(x)
        return x
    
def linear_beta_schedule(timesteps):
    beta_start=0.001
    beta_end=0.02
    return torch.linspace(beta_start, beta_end, timesteps)

timesteps=1001
betas = linear_beta_schedule(timesteps)

alphas = 1-betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1)
sqrt_recip_alphas = torch.rsqrt(alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape)-1))).to(t.device)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(sqrt_one_minus_alpha_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

def p_losses(denoise_model, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t)
    loss = F.mse_loss(noise, predicted_noise)
    return loss

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(sqrt_one_minus_alpha_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    model_mean = sqrt_recip_alphas_t * ( x - (betas_t * model(x, t) / sqrt_one_minus_alpha_cumprod_t ))
    if t_index==0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t)*noise
    
@torch.no_grad()
def p_sample_loop(model, shape, x_start, denoise_steps):
    timesteps = denoise_steps
    device = x_start.device
    b = shape[0]
    noise = torch.randn_like(x_start)
    img = q_sample(x_start, torch.full((b,), timesteps, device=device, dtype=torch.long), noise)
    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
    return img

@torch.no_grad
def sample(model, shape, x_start, denoise_steps):
    return p_sample_loop(model, shape, x_start, denoise_steps)

class FastDiffNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        dims = config.dims
        time_dim = config.time_dim
        
        self.denoiser = FastUnet(dims=dims, time_dim=time_dim)
        self.noise_steps=config.noise_steps
        self.denoise_steps = config.denoise_steps
        self.auc = BinaryAUROC()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        diffusion_loss=None
        x_recon=None
        if self.training:
            t = torch.randint(0, self.noise_steps, (x.size(0),), device=x.device).long()
            diffusion_loss = p_losses(self.denoiser, x, t)
        else:
            x_recon = sample(self.denoiser, shape=x.shape, x_start=x, denoise_steps=self.denoise_steps)
            x_recon = x_recon.permute(0, 2, 1)
        return diffusion_loss, x_recon

class FastDiffNetLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = FastDiffNet(config)
        self.lr=config.lr

    def training_step(self, batch, batch_index):
        x, _ = batch
        loss = self.get_loss(x, mode='train')
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def get_loss(self, x, mode="train"):
        if mode=="train":
            loss, _ = self.model(x)
            return loss
        elif mode=="test":
            _, rec = self.model(x)
            loss = ((rec-x)**2).mean(dim=(1, 2))
            return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        errors = self.get_loss(x, mode="test")
        self.auc.update(errors, y.int())
    
    def on_test_epoch_end(self):
        auc = self.auc.compute()
        self.auc.reset()
        self.log("auc", auc, prog_bar=True)