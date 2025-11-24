import torch 
import torch.nn as nn
import lightning as L 
from models.scorer import StreamScorer


class ConvLayer(nn.Module):
    def __init__(self, n_features, kernel_size=7):
        super().__init__()
        self.padding = nn.ConstantPad1d((kernel_size-1)//2, 0.)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # bs, ws, in_dim
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1) # bs, ws, in_dim
    
class ConvLayer(nn.Module):
    def __init__(self, n_features, kernel_size=7):
        super().__init__()
        self.padding = nn.ConstantPad1d((kernel_size-1)//2, 0.)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # bs, ws, in_dim
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1) # bs, ws, in_dim
    
class GRULayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super().__init__()
        dropout=0 if n_layers==1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
    
    def forward(self, x):
        _, h = self.gru(x)
        return h[-1, :, :]
    
class RNNDecoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super().__init__()
        dropout=0 if n_layers==1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return out
    
class ReconstructionModel(nn.Module):
    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super().__init__()
        self.window_size=window_size
        self.decoder = RNNDecoder(in_dim=in_dim, hid_dim=hid_dim, n_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, out_dim)
        
    def forward(self, x):
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
        decoder_out = self.decoder(h_end_rep)
        out = self.fc_out(decoder_out)
        return out
    
class ForecastingModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super().__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers-1):
            layers.append(nn.Linear(hid_dim, hid_dim))
        layers.append(nn.Linear(hid_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)
    
class FeatureAttentionLayer(nn.Module):
    def __init__(self, in_dim, window_size, dropout, alpha, embed_dim=None, use_bias=True):
        super().__init__()

        self.n_features = self.n_nodes = in_dim
        self.window_size = window_size
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_bias=use_bias
        self.dropout = dropout

        self.embed_dim*=2
        lin_input_dim = 2*window_size
        a_input_dim = self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(in_dim, in_dim))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        a_input = self._make_attention_input(x)
        a_input = self.lin(a_input)
        e = torch.matmul(a_input, self.a).squeeze(3)
        if self.use_bias: e+=self.bias
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, self.training)
        h = self.sigmoid(torch.matmul(attention, x))
        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        K = self.n_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)
        blocks_alternating = v.repeat(1, K, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)
        combined = combined.view(v.size(0), K, K, 2*self.window_size)
        return combined
    
class TemporalAttentionLayer(nn.Module):
    def __init__(self, in_dim, window_size, dropout, alpha, embed_dim=None, use_bias=True):
        super().__init__()

        self.n_features = in_dim
        self.window_size = self.n_nodes = window_size
        self.embed_dim = embed_dim if embed_dim is not None else in_dim
        self.use_bias=use_bias
        self.dropout = dropout

        self.embed_dim*=2
        lin_input_dim = 2*in_dim
        a_input_dim = self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(window_size, window_size))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a_input = self._make_attention_input(x)
        a_input = self.lin(a_input)
        e = torch.matmul(a_input, self.a).squeeze(3)
        if self.use_bias: e+=self.bias
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, self.training)
        h = self.sigmoid(torch.matmul(attention, x))
        return h

    def _make_attention_input(self, v):
        K = self.n_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)
        blocks_alternating = v.repeat(1, K, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)
        combined = combined.view(v.size(0), K, K, 2*self.n_features)
        return combined

class MDAT_GAT(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_dim = out_dim = config.in_dim
        window_size = config.ws


        self.mode=config.mode
        assert self.mode in ["forecast", "recon"], f"Invalid mode: {self.mode}. Choose between 'forecast' and 'recon'."

        kernel_size = 7
        gru_n_layers = 1
        gru_hid_dim = 150
        fc_n_layers = 3
        fc_hid_dim = 150
        recon_n_layers = 1
        recon_hid_dim = 150
        alpha = 0.2
        dropout = 0.2

        self.conv = ConvLayer(n_features=in_dim, kernel_size=kernel_size)
        self.feature_gat = FeatureAttentionLayer(in_dim=in_dim, window_size=window_size, dropout=dropout, alpha=alpha)
        self.temporal_gat = TemporalAttentionLayer(in_dim=in_dim, window_size=window_size, dropout=dropout, alpha=alpha)
        self.gru = GRULayer(in_dim=3*in_dim, hid_dim=gru_hid_dim, n_layers=gru_n_layers, dropout=dropout)

        self.forecasting_model = ForecastingModel(in_dim=gru_hid_dim, hid_dim=fc_hid_dim, out_dim=out_dim, n_layers=fc_n_layers, dropout=dropout)
        self.recon_model = ReconstructionModel(window_size=window_size, in_dim=gru_hid_dim, hid_dim=recon_hid_dim, out_dim=out_dim, n_layers=recon_n_layers, dropout=dropout)

    def forward(self, x): # bs, ws, in_dim
        x = self.conv(x)  # bs, ws, in_dim
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        h_cat = torch.concat([x, h_feat, h_temp], dim=2)
        h_end = self.gru(h_cat)

        if self.mode=="forecast":
            pred = self.forecasting_model(h_end)
            return pred
        elif self.mode=="recon":
            recons = self.recon_model(h_end)
            return recons
        

class MDAT_GAT_Lit(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        if config.mode=="recon":
            config.ws += 1

        self.model = MDAT_GAT(config)
        self.lr = config.lr
        self.criterion = nn.MSELoss(reduction="none")
        self.cri_mode = config.mode
        assert self.cri_mode in ["forecast", "recon"], f"Invalid mode: {self.cri_mode}. Choose between 'forecast' and 'recon'."
        self.scorer = StreamScorer(config.metrics)
        
    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.get_loss(x)
        self.log("train_loss", loss)
        return loss    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def get_loss(self, x, mode=None):

        if self.cri_mode=="forecast":
            inputs = x[:,:-1,:]
            target = x[:,-1,:]
            pred = self.model(inputs)
            loss = self.criterion(pred, target)
            if self.training:
                loss = loss.mean()
            else:
                loss = loss.flatten(start_dim=1).mean(dim=1)
            return loss

        elif self.cri_mode=="recon":
            rec = self.model(x)
            loss = self.criterion(rec, x)
            return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        errors = self.get_loss(x, mode="test")
        self.scorer.update(errors, y.int())
    
    def on_test_epoch_end(self):
        metrics = self.scorer.compute()
        self.scorer.reset()
        for k, v in metrics.items():
            self.log(f"test_{k}", v, prog_bar=True)