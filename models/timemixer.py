import torch 
import torch.nn as nn 
import torch.optim as optim
import lightning as L 
from models.auc import StreamAUC

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:,0:1, :].repeat(1, (self.kernel_size-1)//2, 1)
        end = x[:,-1:, :].repeat(1, (self.kernel_size-1)//2, 1)
        x = torch.cat((front, x, end), dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
    
class series_decomp(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, bias=False, padding_mode='circular')
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
    
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.freq = 4
        self.embed = nn.Linear(self.freq, d_model, bias=False)
    def forward(self, x):
        return self.embed(x)
    
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model)
        self.dp = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dp(x)
    
class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
    
class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, seq_len, down_factor, num_reduce):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_factor**i), seq_len // (down_factor**(i+1))
                        ),
                        nn.GELU(),
                        nn.Linear(
                            seq_len // (down_factor**(i+1)),
                            seq_len // (down_factor**(i+1))
                        )
                    )
                for i in range(num_reduce)
            ]
        )
    
    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i+2 <=len(season_list) -1:
                out_low = season_list[i+2]
            out_season_list.append(out_high.permute(0, 2, 1))
        return out_season_list
    
class MultiScaleTrendMixing(nn.Module):
    def __init__(self, seq_len, down_factor, num_reduce):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_factor**(i+1)), seq_len // (down_factor**(i))
                        ),
                        nn.GELU(),
                        nn.Linear(
                            seq_len // (down_factor**(i)),
                            seq_len // (down_factor**(i))
                        )
                    )
                for i in reversed(range(num_reduce))
            ]
        )
    
    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()

        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i+2 <=len(trend_list_reverse) -1:
                out_high = trend_list_reverse[i+2]
            out_trend_list.append(out_low.permute(0, 2, 1))
        out_trend_list.reverse()
        return out_trend_list
    
class PastDecomposableMixing(nn.Module):
    def __init__(self, seq_len, down_factor, num_reduce, d_model, dropout, d_ff, kernel_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dp = nn.Dropout(dropout)
        self.decomposition = series_decomp(kernel_size=kernel_size)

        self.cross_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(seq_len=seq_len, down_factor=down_factor, num_reduce=num_reduce)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(seq_len=seq_len, down_factor=down_factor, num_reduce=num_reduce)

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)
        
        season_list, trend_list = [], []
        for x in x_list:
            season, trend = self.decomposition(x)
            season = self.cross_layer(season)
            trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for out_season, out_trend, length in zip(out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            out_list.append(out[:,:length, :])
        return out_list
    
class TimeMixer(nn.Module):
    def __init__(self, seq_len, in_dim, num_reduce, down_factor, n_layers, kernel_size, d_model, d_ff, dropout):
        super().__init__()

        self.in_dim = in_dim
        self.n_layers = n_layers
        self.down_factor = down_factor
        self.num_reduce = num_reduce

        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(seq_len=seq_len, down_factor=down_factor, 
                                                    num_reduce=num_reduce, d_model=d_model, d_ff=d_ff, 
                                                    kernel_size=kernel_size, dropout=dropout)
            for _ in range(n_layers)])
        self.preprocess = series_decomp(kernel_size=kernel_size)
        self.enc_embedding = DataEmbedding_wo_pos(c_in=in_dim, d_model=d_model, dropout=dropout)

        self.normalize_layers = nn.ModuleList(
            [
                Normalize(num_features=in_dim, affine=True, non_norm=False)
                for _ in range(num_reduce+1)
            ]
        )
        
        self.projection_layer = nn.Linear(d_model, in_dim)
        self.out_res_layers = nn.ModuleList(
            [
                nn.Linear(seq_len // (down_factor**i), seq_len // (down_factor**i))
                for i in range(num_reduce+1)
            ]
        )
        
    def __multi_scale_process_inputs(self, x_enc, x_mark_enc): # dowsampling over time dimension
        down_pool = nn.AvgPool1d(self.down_factor)
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_ori = x_mark_enc

        x_enc_sampling_list, x_mark_sampling_list = [], []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.num_reduce):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_ori[:, ::self.down_factor, :])
                x_mark_enc_ori = x_mark_enc_ori[:, ::self.down_factor, :]
        
        x_enc = x_enc_sampling_list
        if x_mark_enc_ori is not None:
            x_mark_enc = x_mark_sampling_list
        
        return x_enc, x_mark_enc
    
    def pre_enc(self, x_list):
        out1_list = []
        out2_list = []
        for x in x_list:
            x1, x2 = self.preprocess(x)
            out1_list.append(x1)
            out2_list.append(x2)
        return (out1_list, out2_list)
    
    def forward(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = []
        for i, x in enumerate(x_enc):
            x = self.normalize_layers[i](x, 'norm')
            x_list.append(x)

        enc_out_list = []  
        for x in x_list:
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)

        for i in range(self.n_layers):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        
        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, N, T).permute(0, 2, 1)
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out
    
class TimeMixerLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = TimeMixer(seq_len=config.ws + 1,
                               in_dim=config.in_dim,
                               num_reduce=config.num_reduce,
                               down_factor=config.down_factor,
                               n_layers=config.n_layers,
                               kernel_size=config.kernel_size,
                               d_model=config.d_model,
                               d_ff=config.d_ff,
                               dropout=config.dropout)
        self.lr = config.lr
        self.criterion = nn.MSELoss()
        self.auc = StreamAUC()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed = self.model(x)
        loss = self.criterion(x, reconstructed)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def get_loss(self, x, mode=None):
        reconstructed = self.model(x)
        loss = ((reconstructed - x)**2).flatten(start_dim=1).mean(dim=(1))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        errors = self.get_loss(x, mode="test")
        self.auc.update(errors, y)
    
    def on_test_epoch_end(self):
        auc = self.auc.compute()
        self.auc.reset()
        self.log("auc", auc, prog_bar=True)