import torch
import torch.nn as nn 
import lightning as L 
from models.revin import RevIN

class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_dim = config.in_dim
        hidden_size = config.hid_size
        num_layers = config.num_layers
        bidirectional = config.bidir
        revin = config.revin

        self.revin = revin
        if self.revin: self.revin_layer = RevIN(num_features=in_dim, affine=True, subtract_last=False)

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc_out = nn.Linear(in_features=(1+bidirectional)*hidden_size, out_features=in_dim)

    def forward(self, x):

        if self.revin: 
            x = self.revin_layer(x, 'norm')

        output, (hidden, cell) = self.lstm(x)
        output = self.fc_out(output)
        output = output[:, -1, :].unsqueeze(1)
        if self.revin:
            output = self.revin_layer(output, "denorm")
        
        return output
    
class LSTMLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = LSTM(config)
        self.lr = config.lr
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        inputs = x[:,:-1,:]
        target = x[:,-1,:]
        pred = self.model(inputs)
        loss = self.criterion(pred.squeeze(1), target)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def get_loss(self, x, mode=None):
        inputs = x[:,:-1,:]
        target = x[:,-1,:]
        pred = self.model(inputs)
        loss = torch.abs(target - pred.squeeze(1))  
        return loss