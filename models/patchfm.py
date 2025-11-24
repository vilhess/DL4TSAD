import torch 
import torch.nn as nn 
import torch.nn.functional as F
import lightning as L 
from patchfm import Forecaster, PatchFMConfig
from models.scorer import StreamScorer
    
    
class PatchFMLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.model = Forecaster(PatchFMConfig(compile=config.compile))
        self.scorer = StreamScorer(config.metrics)
    
    def training_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        pass
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        ctx = x[:, :-1, :]
        tgt = x[:, -1, :]
        bs, l, d = ctx.shape
        ctx = ctx.permute(0, 2, 1).reshape(bs*d, l)
        pred, _ = self.model.forecast(ctx, forecast_horizon=1, quantiles=[0.5])
        pred = pred.reshape(bs, d)
        errors = ((pred - tgt.cpu())**2).mean(dim=1)
        self.scorer.update(errors, y.int())
    
    def on_test_epoch_end(self):
        metrics = self.scorer.compute()
        self.scorer.reset()
        for k, v in metrics.items():
            self.log(f"test_{k}", v, prog_bar=True)