import torch
import torch.nn as nn 
import lightning as L 
from models.auc import StreamAUC


class DOC(nn.Module):
    def __init__(self, config):
        super(DOC, self).__init__()
        input_dim = config.in_dim
        hidden_size = config.hid_size
        latent_dim = config.latent_dim
        num_layers = config.num_layers
        bidirectional = config.bidir
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, bias=False)
        self.in_dim = (1+bidirectional)*hidden_size
        self.fc_out = nn.Linear(in_features=self.in_dim, out_features=latent_dim, bias=False)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)
        output = self.fc_out(output)

        return output[:, -1, :]
    
class DOCLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = DOC(config)
        self.lr = config.lr
        self.wd = config.wd
        self.latent_dim = config.latent_dim
        self.center=None
        self.auc = StreamAUC()

    def init_center(self, trainloader, device):
        self.model = self.model.to(device)
        self.eval()

        n_samples = 0
        eps=0.1
        c = torch.zeros(self.latent_dim, device=device)

        with torch.no_grad():
            for x, _ in trainloader:
                x = x.to(device)
                proj = self.model(x)
                n_samples += proj.shape[0]
                c += torch.sum(proj, dim=0)
        c /= n_samples

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.center = c

        self.train()
        return 

    def training_step(self, batch, batch_idx):
        x, _ = batch
        proj = self.model(x)
        dist = torch.sum((proj - self.center.to(proj.device)) ** 2, dim=1)
        loss = torch.mean(dist)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def get_loss(self, x, mode=None):
        proj = self.model(x)
        dist = torch.sum((proj - self.center.to(proj.device)) ** 2, dim=1)
        return dist
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        errors = self.get_loss(x, mode="test")
        self.auc.update(errors, y.int())
    
    def on_test_epoch_end(self):
        auc = self.auc.compute()
        self.auc.reset()
        self.log("auc", auc, prog_bar=True)