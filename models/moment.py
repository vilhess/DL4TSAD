import torch 
import torch.nn as nn 
import torch.nn.functional as F
import lightning as L 
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking
from models.auc import StreamAUC
    
class MomentAD(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        window_size = 512
        config["ws"] = window_size

        self.model = MOMENTPipeline.from_pretrained(
            f"AutonLab/MOMENT-1-{config.size}", 
            model_kwargs={
                'task_name': 'reconstruction',
                'n_channels': config.in_dim,

                'freeze_encoder': config.freeze_encoder,
                'freeze_embedder': config.freeze_encoder,
                'freeze_head': config.freeze_head
            },
        )
        self.model.init()

        self.masking = Masking(mask_ratio=0.3)

    def forward(self, x_enc, input_mask=None, mask=None):
        out = self.model(x_enc=x_enc, input_mask=input_mask, mask=mask).reconstruction
        return out
    
    def get_loss(self, x, mode='train'):

        batch_x = x.permute(0, 2, 1)
        bs, d, l = batch_x.shape
        batch_masks = torch.ones(bs, l).to(batch_x.device)

        if mode == "train":

            batch_x = batch_x.reshape(bs*d, 1, l)
            batch_masks = batch_masks.repeat_interleave(d, dim=0)
            masks = self.masking.generate_mask(x=batch_x, input_mask=batch_masks).to(batch_x.device).long()

            output = self(x_enc=batch_x, input_mask=batch_masks, mask=masks)

            batch_x = batch_x.reshape(bs, d, l)
            output = output.reshape(bs, d, l)

        elif mode == "test":
            output = self(x_enc=batch_x, input_mask=batch_masks)

        error = ((output - batch_x)**2).flatten(start_dim=1).mean(dim=(1))
        return error
    
class MomentLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = MomentAD(config)
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