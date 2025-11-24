import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L 
from models.scorer import StreamScorer


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        input_dim = config.in_dim
        hidden_size = config.hid_size
        num_layers = config.num_layers
        bidirectional = config.bidir

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc_out = nn.Linear(in_features=(1+bidirectional)*hidden_size, out_features=1)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)
        output = self.fc_out(output)
        return output[:, -1, :]
    
class DROCCLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = LSTM(config)
        self.lr = config.lr
        self.lamda = config.lam
        self.radius = config.radius
        self.gamma = config.gamma
        self.asc_num_steps = config.num_steps
        self.asc_step_size = config.step_size
        self.only_ce_epochs = config.ce_epochs
        self.epochs = config.epochs
        self.scorer = StreamScorer(config.metrics)

    def training_step(self, batch, batch_idx):
        x, target = batch
        target = torch.squeeze(target)
        logits = self.model(x)
        logits = torch.squeeze(logits, dim = 1)
        target = target.float()
        ce_loss = F.binary_cross_entropy_with_logits(logits, target)
        if self.current_epoch >= self.only_ce_epochs:
            x = x.detach()
            x = x[target==0]
            adv_loss = self.one_class_adv_loss(x)
            loss = ce_loss + adv_loss * self.lamda
        else:
            loss = ce_loss
        self.log('train_loss', loss)
        self.log('ce_loss', ce_loss)
        self.log('adv_loss', adv_loss)
        return loss
    
    def get_loss(self, x, mode=None):
        return torch.sigmoid(self.model(x))
    
    def one_class_adv_loss(self, x_train_data):
        batch_size = len(x_train_data)
        x_adv = torch.randn_like(x_train_data[:, :, 0], device=self.device).detach().requires_grad_()
        x_adv_sampled = x_train_data.clone()
        x_adv_sampled[:, :, 0] = x_train_data[:, :, 0] + x_adv

        for step in range(self.asc_num_steps):
            with torch.enable_grad():

                new_targets = torch.zeros(batch_size, 1).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)
                
                logits = self.model(x_adv_sampled)     
  
                logits = torch.squeeze(logits, dim = 1)
                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim = tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1]*(grad.dim()-1))
                grad_normalized = grad/grad_norm 
            with torch.no_grad():
                x_adv_sampled.add_(self.asc_step_size * grad_normalized)

            if (step + 1) % 10==0:
                h = x_adv_sampled - x_train_data
                norm_h = torch.sqrt(torch.sum(h**2, 
                                                dim=tuple(range(1, h.dim()))))
                alpha = torch.clamp(norm_h, self.radius, 
                                    self.gamma * self.radius).to(self.device)
                proj = (alpha/norm_h).view(-1, *[1] * (h.dim()-1))
                h = proj * h
                x_adv_sampled = x_train_data + h 

        adv_pred = self.model(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets+1))

        return adv_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def adjust_learning_rate(self, epoch, total_epochs, only_ce_epochs, learning_rate, optimizer):
        epoch = epoch - only_ce_epochs
        drocc_epochs = total_epochs - only_ce_epochs
        if epoch <= drocc_epochs:
            lr = learning_rate * 0.1
        if epoch <= 0.50 * drocc_epochs:
            lr = learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer
    
    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        current_epoch = self.current_epoch
        initial_lr = self.lr
        self.adjust_learning_rate(optimizer=optimizer, total_epochs=self.epochs, epoch=current_epoch, only_ce_epochs=self.only_ce_epochs, learning_rate=initial_lr)

    def test_step(self, batch, batch_idx):
        x, y = batch
        errors = self.get_loss(x, mode="test")
        self.scorer.update(errors, y.int())
    
    def on_test_epoch_end(self):
        metrics = self.scorer.compute()
        self.scorer.reset()
        for k, v in metrics.items():
            self.log(f"test_{k}", v, prog_bar=True)