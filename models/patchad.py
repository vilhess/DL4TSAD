import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from einops import rearrange, repeat
from tkinter import _flatten
from einops.layers.torch import Rearrange
from einops import reduce
import lightning as L 
from models.scorer import StreamScorer


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = torch.ones(self.num_features)
        self.affine_weight = torch.nn.Parameter(self.affine_weight, requires_grad=False)
        self.affine_bias = torch.zeros(self.num_features)
        self.affine_bias = torch.nn.Parameter(self.affine_bias, requires_grad=False)
        # self.affine_weight=self.affine_weight.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        # self.affine_bias=self.affine_bias.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    




class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        # L, 1
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # 
        if d_model % 2 == 0:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

# class PositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEmbedding, self).__init__()
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False

#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.00):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        # return self.dropout(x)
        return x
    

class ChInd_PositionalEmbedding(nn.Module):
    def __init__(self, max_len=5000):
        super(ChInd_PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(1, max_len, 1).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(0).unsqueeze(-1)
        # L, 1
        # 
        # pe += torch.sin(position)
        pe += position

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class ChInd_DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.00):
        super(ChInd_DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = ChInd_PositionalEmbedding()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class ChInd_DataEmbedding2(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.00):
        super(ChInd_DataEmbedding2, self).__init__()

        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = ChInd_PositionalEmbedding()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    




def get_activation(activ):
    if activ == "relu":
        return nn.ReLU()
    elif activ == "gelu":
        return nn.GELU()
    elif activ == "leaky_relu":
        return nn.LeakyReLU()
    elif activ == 'tanh':
        return nn.Tanh()
    elif activ == 'sigmoid':
        return nn.Sigmoid()
    elif activ == "none":
        return nn.Identity()
    else:
        raise ValueError(f"activation:{activ}")

def get_norm(norm, c):
    if norm == 'bn':
        norm_class = nn.BatchNorm2d(c)
    elif norm == 'in':
        norm_class = nn.InstanceNorm2d(c)
    elif norm == 'ln':
        norm_class = nn.LayerNorm(c)
    else:
        norm_class = nn.Identity()

    return norm_class


class MLPBlock(nn.Module):
    def __init__(
        self,
        dim,
        in_features: int,
        hid_features: int,
        out_features: int,
        activ="gelu",
        drop: float = 0.00,
        jump_conn="proj",
        norm='ln'
    ):
        super().__init__()
        self.dim = dim
        self.out_features = out_features
        norm
        self.net = nn.Sequential(
            get_norm(norm,in_features),
            nn.Linear(in_features, hid_features),
            get_activation(activ),
            get_norm(norm,hid_features),
            nn.Linear(hid_features, out_features),
            nn.Dropout(drop),
        )
        if jump_conn == "trunc":
            self.jump_net = nn.Identity()
        elif jump_conn == "proj":
            self.jump_net = nn.Linear(in_features, out_features)
        else:
            raise ValueError(f"jump_conn:{jump_conn}")

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        x = self.jump_net(x) + self.net(x)
        x = torch.transpose(x, self.dim, -1)
        return x
    


class PatchMLP_layer(nn.Module):
    def __init__(
        self,
        in_len: int,
        hid_len: int,

        in_chn: int,
        hid_chn: int,
        out_chn,

        patch_size: int,
        hid_pch: int,

        d_model: int,
        norm=None,
        activ="gelu",
        drop: float = 0.0,
        jump_conn='proj'
    ) -> None:
        super().__init__()
        # B C N P
        self.ch_mixing1 = MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop, jump_conn=jump_conn)
        # self.ch_mixing2 = MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop, jump_conn=jump_conn)
        self.patch_num_mix = MLPBlock(2, in_len // patch_size, hid_len, in_len // patch_size, activ, drop, jump_conn=jump_conn)
        self.patch_size_mix = MLPBlock(2, patch_size, hid_pch, patch_size, activ, drop,jump_conn=jump_conn)
        self.d_mixing1 = MLPBlock(3, d_model, d_model, d_model, activ, drop, jump_conn=jump_conn)

        if norm == 'bn':
            norm_class = nn.BatchNorm2d
        elif norm == 'in':
            norm_class = nn.InstanceNorm2d
        elif norm == 'ln':
            norm_class = nn.LayerNorm
        else:
            norm_class = nn.Identity
        self.norm1 = norm_class(in_chn)
        self.norm2 = norm_class(out_chn)

    def forward(self, x_patch_num, x_patch_size):
        # B C N P
        x_patch_num = self.norm1(x_patch_num)
        x_patch_num = self.ch_mixing1(x_patch_num)
        x_patch_num = self.norm2(x_patch_num)
        x_patch_num = self.patch_num_mix(x_patch_num)
        x_patch_num = self.norm2(x_patch_num)
        x_patch_num = self.d_mixing1(x_patch_num)

        x_patch_size = self.norm1(x_patch_size)
        x_patch_size = self.ch_mixing1(x_patch_size)
        x_patch_size = self.norm2(x_patch_size)
        x_patch_size = self.patch_size_mix(x_patch_size)
        x_patch_size = self.norm2(x_patch_size)
        x_patch_size = self.d_mixing1(x_patch_size)

        return x_patch_num, x_patch_size


class Encoder(nn.Module):
    def __init__(self, enc_layers):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList(enc_layers)
        self.num_mix_layer = nn.Sequential(nn.Linear(len(enc_layers), len(enc_layers)*2), nn.Sigmoid(), nn.Linear(len(enc_layers)*2,1), nn.Sigmoid(), Rearrange('b n p k -> b (n p) k'))
        self.size_mix_layer = nn.Sequential(nn.Linear(len(enc_layers), len(enc_layers)*2), nn.Sigmoid(), nn.Linear(len(enc_layers)*2,1) , nn.Sigmoid(), Rearrange('b n p k -> b (n p) k'))
        self.softmax = nn.Softmax(-1)


    def forward(self, x_patch_num, x_patch_size, mask=None):
        num_dist_list = []
        size_dist_list = []
        num_logi_list = []
        size_logi_list = []

        for enc in self.enc_layers:
            x_pach_num_dist, x_patch_size_dist = enc(x_patch_num, x_patch_size)

            num_logi_list.append(x_pach_num_dist.mean(1))
            size_logi_list.append(x_patch_size_dist.mean(1))


            x_pach_num_dist = self.softmax(x_pach_num_dist)
            x_patch_size_dist = self.softmax(x_patch_size_dist)

            x_pach_num_dist = reduce(
                x_pach_num_dist,
                "b reduce_c n d -> b n d",
                "mean",
            )
            x_patch_size_dist = reduce(
                x_patch_size_dist,
                "b reduce_c n d -> b n d",
                "mean",
            )
            
            x_pach_num_dist = rearrange(x_pach_num_dist, "b n p -> b (n p) 1")
            x_patch_size_dist = rearrange(x_patch_size_dist, "b p n -> b (p n) 1")

            num_dist_list.append(x_pach_num_dist)
            size_dist_list.append(x_patch_size_dist)

        return num_dist_list, size_dist_list, num_logi_list, size_logi_list
    

class Ensemble_block(nn.Module):
    def __init__(self, e_layers) -> None:
        super().__init__()
        self.mix_layer = nn.parameter.Parameter(torch.ones(e_layers), requires_grad=True)
        pass
    
    def forward(self, dist_list):
        # list of B N D
        dist_list = torch.stack(dist_list, dim=-1)

        # Apply softmax to the mix_layer weights
        weights = torch.softmax(self.mix_layer, dim=0)

        # Apply the weights to dist_list
        dist_list = dist_list * weights

        dist_list = torch.split(dist_list, 1, dim=3)
        dist_list = [t.squeeze(3) for t in dist_list]

        return dist_list

class Mean_Ensemble_block(nn.Module):
    def __init__(self, e_layers) -> None:
        super().__init__()
        pass

    def forward(self, dist_list):
        dist_list = torch.stack(dist_list, dim=-1).mean(-1,keepdim=False)

        return [dist_list]

        


class Encoder_Ensemble(nn.Module):
    def __init__(self, enc_layers ):
        super(Encoder_Ensemble, self).__init__()
        self.enc_layers = nn.ModuleList(enc_layers)
 

        self.num_mix_layer = Ensemble_block(len(enc_layers))
        self.size_mix_layer = Ensemble_block(len(enc_layers))


        self.softmax = nn.Softmax(-1)


    def forward(self, x_patch_num, x_patch_size, mask=None):
        num_dist_list = []
        size_dist_list = []
        num_logi_list = []
        size_logi_list = []
        T_num_logi_list =[]
        T_size_logi_list = []

        for enc in self.enc_layers:
            x_pach_num_dist, x_patch_size_dist = enc(x_patch_num, x_patch_size)

            x_patch_num = torch.relu(x_patch_num)
            x_patch_size = torch.relu(x_patch_size)

            T_num_logi_list.append(x_pach_num_dist)
            T_size_logi_list.append(x_patch_size_dist)

            num_logi_list.append(x_pach_num_dist.mean(1))
            size_logi_list.append(x_patch_size_dist.mean(1))


            x_pach_num_dist = self.softmax(x_pach_num_dist)
            x_patch_size_dist = self.softmax(x_patch_size_dist)

            x_pach_num_dist = reduce(
                x_pach_num_dist,
                "b reduce_c n d -> b n d",
                "mean",
            )
            x_patch_size_dist = reduce(
                x_patch_size_dist,
                "b reduce_c n d -> b n d",
                "mean",
            )
            
            num_dist_list.append(x_pach_num_dist)
            size_dist_list.append(x_patch_size_dist)

            
        num_dist_list = self.num_mix_layer(num_dist_list)
        size_dist_list = self.size_mix_layer(size_dist_list)
        

        return num_dist_list, size_dist_list, num_logi_list, size_logi_list, T_num_logi_list, T_size_logi_list



class PatchMLPAD(nn.Module):
    def __init__(self, config):
        super().__init__()

        win_size = config.ws+1
        d_model = config.d_model
        expand_ratio = 1.2
        e_layer = config.e_layers
        patch_sizes = config.patch_sizes
        dropout = config.dp
        activation = "gelu"
        channel = config.in_dim
        cont_model = config.ws+1
        norm = "n"
        output_attention = config.out_attn
        self.patch_sizes = patch_sizes
        self.win_size = win_size
        self.output_attention = output_attention

        self.win_emb = PositionalEmbedding(channel)

        self.patch_num_emb = nn.ModuleList(
            [
                nn.Linear(patch_size,d_model) for patch_size in patch_sizes
            ]
        )
        self.patch_size_emb = nn.ModuleList(
            [
                nn.Linear(win_size//patch_size, d_model) for patch_size in patch_sizes
            ]
        )
        self.patch_encoders = nn.ModuleList()
        cont_model = d_model if cont_model is None else cont_model
        cont_model = 30
        

        self.patch_num_mixer = nn.Sequential(MLPBlock(2, d_model, d_model//2, d_model, activ=activation, drop=dropout, jump_conn='trunc'),nn.Softmax(-1))
        self.patch_size_mixer = nn.Sequential(MLPBlock(2, d_model, d_model//2, d_model, activ=activation, drop=dropout, jump_conn='trunc'),nn.Softmax(-1))


        for i, p in enumerate(patch_sizes):
            # Multi patch
            patch_size = patch_sizes[i]
            patch_num = win_size // patch_size
            enc_layers = [
                PatchMLP_layer(win_size, 40, channel, int(channel*1.2), int(channel*1.), patch_size, int(patch_size*1.2), d_model, norm, activation, dropout, jump_conn='proj')
                for i in range(e_layer)
            ]
            enc = Encoder_Ensemble(enc_layers=enc_layers)
            self.patch_encoders.append(enc)

        self.recons_num = []
        self.recons_size = []
        for i, p in enumerate(patch_sizes):
            patch_size = patch_sizes[i]
            patch_num = win_size // patch_size
            self.recons_num.append(nn.Sequential(Rearrange('b c n p -> b c (n p)'), nn.LayerNorm(patch_num*d_model),  nn.Linear(patch_num*d_model, d_model), nn.Sigmoid(), nn.LayerNorm(d_model), nn.Linear(d_model, win_size), Rearrange('b c l -> b l c')))

            self.recons_size.append(nn.Sequential(Rearrange('b c n p -> b c (n p)'), nn.LayerNorm(patch_size*d_model),  nn.Linear(patch_size*d_model, d_model), nn.Sigmoid(), nn.LayerNorm(d_model), nn.Linear(d_model, win_size), Rearrange('b c l -> b l c')))

        self.recons_num = nn.ModuleList(self.recons_num)
        self.recons_size = nn.ModuleList(self.recons_size)

        self.rec_alpha = nn.Parameter(torch.zeros(patch_size), requires_grad=True)
        self.rec_alpha.data.fill_(0.5)

    def forward(self, x, mask=None, del_inter=0, del_intra=0):
        B, L, M = x.shape  # Batch win_size channel
        patch_num_distribution_list = []
        patch_size_distribution_list = []
        patch_num_mx_list = []
        patch_size_mx_list = []
        mask_patch_num_list = []
        mask_patch_size_list = []
        revin_layer = RevIN(num_features=M).to(x.device)

        # Instance Normalization Operation
        x = revin_layer(x, "norm")

        rec_x = None
        # Mutil-scale Patching Operation
        for patch_index, patchsize in enumerate(self.patch_sizes):
            patch_enc = self.patch_encoders[patch_index]
            x = x + self.win_emb(x)
            # x = self.win_emb(x)
            x_patch_num = x_patch_size = x
            # B L C

            x_patch_num = rearrange(x_patch_num, "b (n p) c -> b c n p", p=patchsize)
            x_patch_size = rearrange(x_patch_size, "b (p n) c-> b c p n", p=patchsize)

            x_patch_num = self.patch_num_emb[patch_index](x_patch_num)
            x_patch_size = self.patch_size_emb[patch_index](x_patch_size)

            # B C N D
            (
                patch_num_distribution,
                patch_size_distribution,
                logi_patch_num,
                logi_patch_size,
                T_num_logi_list,
                T_size_logi_list
            ) = patch_enc(x_patch_num, x_patch_size, mask)

            patch_num_distribution_list.append(patch_num_distribution)
            patch_size_distribution_list.append(patch_size_distribution)


            recs = []
            for i in range(len(logi_patch_num)):
                logi_patch_num1 = logi_patch_num[i]
                logi_patch_size1 = logi_patch_size[i]
                patch_num_mx = self.patch_num_mixer(logi_patch_num1)
                patch_size_mx = self.patch_size_mixer(logi_patch_size1)
                patch_num_mx_list.append(patch_num_mx)
                patch_size_mx_list.append(patch_size_mx)


                # print(len(T_num_logi_list))
                # print(T_num_logi_list[i].shape)
                rec1 = self.recons_num[patch_index](T_num_logi_list[i])
                rec2 = self.recons_size[patch_index](T_size_logi_list[i])

                if del_inter:
                    rec = rec2
                elif del_intra:
                    rec = rec1
                else:
                    rec_alpha = self.rec_alpha[patch_index]
                    rec = rec1 * rec_alpha + rec2 * (1 - rec_alpha)
                recs.append(rec)

            recs = torch.stack(recs, dim=0).mean(0)

            if not self.training:
                # self.T1 = torch.stack(T_num_logi_list, dim=0).mean(0)
                # self.T2 = torch.stack(T_size_logi_list, dim=0).mean(0)
                self.T1 = T_num_logi_list[-1]
                self.T2 = T_size_logi_list[-1]
            
            if rec_x is None:
                rec_x = recs
            else:
                rec_x = rec_x + recs

        rec_x = rec_x / len(self.patch_sizes)    
        # rec_x = revin_layer(x, 'denorm')



        patch_num_distribution_list = list(_flatten(patch_num_distribution_list))
        patch_size_distribution_list = list(_flatten(patch_size_distribution_list))
        patch_num_mx_list = list(_flatten(patch_num_mx_list))
        patch_size_mx_list = list(_flatten(patch_size_mx_list))

        if self.output_attention:
            return (
                patch_num_distribution_list,
                patch_size_distribution_list,
                patch_num_mx_list,
                patch_size_mx_list,
                rec_x
            )
        else:
            return None
        

def my_kl_loss(p, q):
    # B N D
    res = p * (torch.log(p + 0.0000001) - torch.log(q + 0.0000001))
    # B N
    return torch.sum(res, dim=-1)


def inter_intra_dist(p,q,w_de=True,train=1,temp=1):
    # B N D
    if train:
        if w_de:
            p_loss = torch.mean(my_kl_loss(p,q.detach()*temp)) + torch.mean(my_kl_loss(q.detach(),p*temp))
            q_loss = torch.mean(my_kl_loss(p.detach(),q*temp)) + torch.mean(my_kl_loss(q,p.detach()*temp))
        else:
            p_loss = -torch.mean(my_kl_loss(p,q.detach())) 
            q_loss = -torch.mean(my_kl_loss(q,p.detach())) 
    else:
        if w_de:
            p_loss = my_kl_loss(p,q.detach()) + my_kl_loss(q.detach(),p)
            q_loss = my_kl_loss(p.detach(),q) + my_kl_loss(q,p.detach())

        else:
            p_loss = -(my_kl_loss(p,q.detach())) 
            q_loss = -(my_kl_loss(q,p.detach())) 

    return p_loss,q_loss


def normalize_tensor(tensor):
    # tensor: B N D
    sum_tensor = torch.sum(tensor,dim=-1,keepdim=True)
    normalized_tensor = tensor / sum_tensor
    return normalized_tensor

def anomaly_score(patch_num_dist_list,patch_size_dist_list, win_size, train=1, temp=1, w_de=True):
    for i in range(len(patch_num_dist_list)):
        patch_num_dist = patch_num_dist_list[i]
        patch_size_dist = patch_size_dist_list[i]


        patch_num_dist = repeat(patch_num_dist,'b n d -> b (n rp) d',rp=win_size//patch_num_dist.shape[1])
        patch_size_dist = repeat(patch_size_dist,'b p d -> b (rp p) d',rp=win_size//patch_size_dist.shape[1])

        patch_num_dist = normalize_tensor(patch_num_dist)
        patch_size_dist = normalize_tensor(patch_size_dist)

        patch_num_loss,patch_size_loss = inter_intra_dist(patch_num_dist,patch_size_dist,w_de,train=train,temp=temp)

        if i==0:
            patch_num_loss_all = patch_num_loss
            patch_size_loss_all = patch_size_loss
        else:
            patch_num_loss_all += patch_num_loss
            patch_size_loss_all += patch_size_loss

    return patch_num_loss_all,patch_size_loss_all

class PatchADLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = PatchMLPAD(config)
        self.lr = config.lr
        self.criterion = nn.MSELoss()
        self.criterion_wise = nn.MSELoss(reduction='none')
        self.patch_mx = config.patch_mx
        self.beta = config.beta
        self.ws = config.ws+1
        self.scorer = StreamScorer(config.metrics)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list,recx = self.model(x)
        loss = 0.
        cont_loss1, cont_loss2 = anomaly_score(patch_num_dist_list, patch_size_mx_list, win_size=self.ws, train=1, temp=1)
        cont_loss_1 = cont_loss1 - cont_loss2
        loss-= self.patch_mx * cont_loss1

        cont_loss12, cont_loss22 = anomaly_score(patch_num_mx_list,patch_size_dist_list,win_size=self.ws,train=1,temp=1)
        cont_loss_2 = cont_loss12 - cont_loss22
        loss-= self.patch_mx * cont_loss2

        patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=self.ws,train=1,temp=1)
        patch_num_loss = patch_num_loss / len(patch_num_dist_list)
        patch_size_loss = patch_size_loss / len(patch_num_dist_list)

        loss3 = patch_num_loss - patch_size_loss
        loss -= loss3 * (1-self.patch_mx)

        loss_mse = self.criterion(recx, x)
        loss += loss_mse*10  
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def get_loss(self, x, mode=None):
        patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list,recx = self.model(x)
        patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=self.ws,train=0)

        patch_num_loss = patch_num_loss / len(patch_num_dist_list)
        patch_size_loss = patch_size_loss / len(patch_num_dist_list)

        loss3 = patch_size_loss - patch_num_loss
        mse_loss_ = self.criterion_wise(recx,x)
        metric1 = torch.softmax((-patch_num_loss), dim=-1)
        metric2 = mse_loss_.mean(-1)

        metric = metric1 * (self.beta) + metric2 * (1-self.beta)
        cri = metric.detach().cpu()
        loss = cri[:, -1]
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