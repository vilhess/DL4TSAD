import torch 
import torch.nn as nn 
from torch.nn import TransformerEncoder, TransformerDecoder
import math
import lightning as L 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, in_dim, nheads, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=nheads, dropout=dropout)
        self.dp = nn.Dropout(dropout)

        self.fc1 = nn.Linear(in_features=in_dim, out_features=dim_feedforward)
        self.dp1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(in_features=dim_feedforward, out_features=in_dim)
        self.dp2 = nn.Dropout(dropout)

        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal=None):
        h1 = self.self_attn(x, x, x)[0]
        h = x + self.dp(h1)
        h1 = self.fc2(self.dp1(self.act(self.fc1(h))))
        h = h + self.dp2(h1)
        return h
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, in_dim, nheads, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=nheads, dropout=dropout)
        self.dp1 = nn.Dropout(dropout)

        self.mha = nn.MultiheadAttention(embed_dim=in_dim, num_heads=nheads, dropout=dropout)
        self.dp2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(in_features=in_dim, out_features=dim_feedforward)
        self.dp3 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(in_features=dim_feedforward, out_features=in_dim)
        self.dp4 = nn.Dropout(dropout)

        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=None):
        h1 = self.self_attn(x, x, x)[0]
        h = self.dp1(h1) + x
        h1 = self.mha(h, memory, memory)[0]
        h = self.dp2(h1) + h
        h1 = self.fc2(self.dp3(self.act(self.fc1(h))))
        h = self.dp4(h1) + h
        return h
    

class TranAD(nn.Module):
    def __init__(self, config):
        super(TranAD, self).__init__()

        in_dim = config.in_dim
        window = config.ws+1
        self.in_dim = in_dim

        self.pos_encoder = PositionalEncoding(d_model=2*in_dim, dropout=0.1, max_len=window)
        encoder_layer = TransformerEncoderLayer(in_dim=2*in_dim, nheads=in_dim, dim_feedforward=16, dropout=0.1)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)
        decoder_layer1 = TransformerDecoderLayer(in_dim=2*in_dim, nheads=in_dim, dim_feedforward=16, dropout=0.1)
        self.decoder1 = TransformerDecoder(decoder_layer=decoder_layer1, num_layers=1)
        decoder_layer2 = TransformerDecoderLayer(in_dim=2*in_dim, nheads=in_dim, dim_feedforward=16, dropout=0.1)
        self.decoder2 = TransformerDecoder(decoder_layer=decoder_layer2, num_layers=1)
        self.fc_out = nn.Linear(in_features=2*in_dim, out_features=in_dim)

    def encode(self, x, c, target):
        x = torch.cat([x, c], dim=2)
        x = x*math.sqrt(self.in_dim)
        x = self.pos_encoder(x)
        memory = self.encoder(x)
        target = target.repeat(1, 1, 2)
        return target, memory
    
    def forward(self, x, target):
        
        # Phase 1:
        c = torch.zeros_like(x)
        target, memory = self.encode(x, c, target)
        x1 = self.decoder1(target, memory)
        x1 = self.fc_out(x1)

        # Phase 2:
        target = target[:, :, :target.size(2)//2]
        c = (x1 - x)**2
        target, memory = self.encode(x, c, target)
        x2 = self.decoder2(target, memory)
        x2 = self.fc_out(x2)
        
        return x1, x2
    
class TranADLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = TranAD(config)
        self.lr = config.lr
        self.criterion = nn.MSELoss(reduction="none")

    def training_step(self, batch, batch_idx):
        epoch = self.trainer.current_epoch
        x, _ = batch
        x = x.permute(1, 0, 2)
        elem = x[-1, :, :].view(1, x.size(1), x.size(2))
        x1, x2 = self.model(x, elem)
        loss = 1/(epoch+1) * self.criterion(elem, x1) + (1 - 1/(epoch+1)) * self.criterion(elem, x2)
        loss = torch.mean(loss)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters() , lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def get_loss(self, x, mode=None):
        x = x.permute(1, 0, 2)
        elem = x[-1, :, :].view(1, x.size(1), x.size(2))
        o1, o2 = self.model(x, elem)
        loss = self.criterion(o2, elem).permute(1, 0, 2)
        loss = torch.mean(loss, dim=(1, 2))
        return loss