from typing import Optional
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
import lightning as L 
from models.auc import StreamAUC

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_ln = False
        self.seq_len = configs.ws
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.d_ff = configs.d_ff
        self.patch_num = (configs.ws - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(configs.in_dim * self.patch_size, configs.d_model, 0.1)

        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layer]
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(0))
            self.gpt2.to(device=device)

        # self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.ln_proj = nn.LayerNorm(configs.d_ff)

        self.out_layer = nn.Linear(
            configs.d_ff, 
            configs.in_dim, 
            bias=True)

    def get_loss(self, x_enc, mode="train"):
        dec_out = self.anomaly_detection(x_enc)
        loss = F.mse_loss(dec_out, x_enc, reduction='none')
        loss = loss.mean(dim=(1, 2)) 
        return loss

    
    def anomaly_detection(self, x_enc):
        B, L, M = x_enc.shape

        seg_num = 25
        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')

        enc_out = torch.nn.functional.pad(x_enc, (0, 768-x_enc.shape[-1]))
        
        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        
        outputs = outputs[:, :, :self.d_ff]
        dec_out = self.out_layer(outputs)

        dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=seg_num)
        dec_out = dec_out * \
                  (stdev[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = dec_out + \
                  (means[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')
        return dec_out


    
class GPT4TSLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = Model(config)
        self.lr = config.lr
        self.auc = StreamAUC()
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.model.get_loss(x, mode="train")
        loss = loss.mean()
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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