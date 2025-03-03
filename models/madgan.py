import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import lightning as L 

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        in_dim = config.in_dim
        latent_dim = config.latent_dim
        hidden_dim = config.hid_dim
        num_layers = config.num_layers
        bidirectional = config.bidir
        
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(in_features=(1+bidirectional)*hidden_dim, out_features=in_dim)
        nn.init.trunc_normal_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight)

    def forward(self, z):
        rnn_out, _ = self.lstm(z) 
        return self.linear(rnn_out)
    
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        in_dim = config.in_dim
        hidden_dim = config.hid_dim
        num_layers = config.num_layers
        bidirectional = config.bidir

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=(1+bidirectional)*hidden_dim, out_features=1)

        nn.init.trunc_normal_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight)

    def forward(self, x):
            
        output, (hidden, cell) = self.lstm(x)
        output = self.linear(output)
        return output[:, -1, :]
    

def get_best_latent(gen, data, latent_dim):
    gen.train()

    max_iters = 50

    Z = torch.randn((data.size(0), data.size(1), latent_dim), requires_grad=True, device=data.device)
    optimizer = optim.RMSprop(params=[Z], lr=0.1)
    loss_fn = nn.MSELoss(reduction="none")

    normalize_target = F.normalize(data, dim=1, p=2)

    for _ in range(max_iters):
        optimizer.zero_grad()
        
        generated_samples = gen(Z)
        normalized = F.normalize(generated_samples, dim=1, p=2)

        reconstruction_loss = loss_fn(normalized, normalize_target)
        reconstruction_loss = reconstruction_loss.sum(dim=(1, 2)).mean()

        reconstruction_loss.backward()
        optimizer.step()
    gen.eval()
    return Z

class MADGANLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.disc = Discriminator(config)
        self.gen = Generator(config)
        self.lr = config.lr
        self.ws = config.ws+1
        self.latent_dim = config.latent_dim
        self.weight = config.weight
        self.automatic_optimization = False
    
    def training_step(self, batch, batch_idx):
        optim_disc, optim_gen = self.configure_optimizers()
        x, _ = batch
        bs = x.size(0)

        optim_disc.zero_grad()
        ones = torch.ones(bs, 1, device=x.device)
        zeros = torch.ones(bs, 1, device=x.device)

        pred_disc_true = self.disc(x)
        loss_disc_true = F.binary_cross_entropy_with_logits(pred_disc_true, zeros)

        z = torch.randn((bs, self.ws, self.latent_dim), device=x.device)
        
        fake = self.gen(z)

        pred_disc_fake = self.disc(fake.detach())
        loss_disc_fake = F.binary_cross_entropy_with_logits(pred_disc_fake, ones)
        self.manual_backward(loss_disc_fake)
        self.manual_backward(loss_disc_true)
        optim_disc.step()

        optim_gen.zero_grad()
        pred_disc_fake = self.disc(fake)
        loss_gen = F.binary_cross_entropy_with_logits(pred_disc_fake, zeros)
        self.manual_backward(loss_gen)
        optim_gen.step()
        self.log(f"train_gen_loss: {loss_gen}; train_disc_loss_fake: {loss_disc_fake}; train_disc_loss_true: {loss_disc_true}")

    
    def configure_optimizers(self):
        optim_disc = torch.optim.Adam(self.disc.parameters(), lr=self.lr)
        optim_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
        return optim_disc, optim_gen
    
    def get_loss(self, x, mode=None):

        with torch.enable_grad():
            batch_latent = get_best_latent(self.gen, x.clone(), latent_dim=self.latent_dim)
        generated = self.gen(batch_latent)
        disc_score = torch.sigmoid(self.disc(generated))
        disc_score = torch.squeeze(disc_score)
        res_loss = (generated - x).abs().sum(dim=(1, 2))
        score = self.weight * disc_score + (1-self.weight)*res_loss
        return score