import torch 
import torch.nn as nn 
import lightning as L 

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_size = config.in_dim * (config.ws+1)
        latent_dim = config.latent_dim

        self.linear1 = nn.Linear(in_size, int(in_size/2))
        self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
        self.linear3 = nn.Linear(int(in_size/4), latent_dim)
        self.relu = nn.ReLU()

    def forward(self, w):
        out = self.relu(self.linear1(w))
        out = self.relu(self.linear2(out))
        z = self.relu(self.linear3(out))
        return z
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        out_size = config.in_dim * (config.ws+1)
        latent_dim = config.latent_dim

        self.linear1 = nn.Linear(latent_dim, int(out_size//4))
        self.linear2 = nn.Linear(int(out_size//4), int(out_size//2))
        self.linear3 = nn.Linear(int(out_size//2), out_size)
        self.relu = nn.ReLU()

    def forward(self, z):
        out = self.relu(self.linear1(z))
        out = self.relu(self.linear2(out))
        out = self.linear3(out)
        return out
    

class USADLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.enc = Encoder(config)
        self.dec1 = Decoder(config)
        self.dec2 = Decoder(config)
        self.lr = config.lr
        self.alpha = config.alpha
        self.beta = config.beta
        self.automatic_optimization = False
    
    def training_step(self, batch, batch_idx):
        optim1, optim2 = self.optimizers()
        x, _ = batch
        x = x.flatten(start_dim=1)

        z = self.enc(x)
        w1 = self.dec1(z)
        w3 = self.dec2(self.enc(w1))
        loss1 = 1/(self.current_epoch+1) * torch.mean((x - w1)**2) + (1 - 1/(self.current_epoch+1)) * torch.mean((x - w3)**2)

        optim1.zero_grad()
        self.manual_backward(loss1)
        optim1.step()

        z = self.enc(x)
        w1 = self.dec1(z)
        w2 = self.dec2(z)
        w3 = self.dec2(self.enc(w1))
        loss2 = 1/(self.current_epoch+1) * torch.mean((x - w2)**2) - (1 - 1/(self.current_epoch+1)) * torch.mean((x - w3)**2)
        optim2.zero_grad()
        self.manual_backward(loss2)
        optim2.step()
        self.log('loss1', loss1)
        self.log('loss2', loss2)

    def configure_optimizers(self):
        optim1 = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec1.parameters()), lr=self.lr)
        optim2 = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec2.parameters()), lr=self.lr)
        return optim1, optim2
    
    def get_loss(self, x, mode=None):
        x = x.flatten(start_dim=1)
        w1 = self.dec1(self.enc(x))
        w2 = self.dec2(self.enc(w1))
        loss = self.alpha * torch.mean((x-w1)**2 + self.beta * torch.mean(x - w2)**2, dim=1)
        return loss