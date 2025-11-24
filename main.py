import torch
import numpy as np
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch import seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf
import gc
from datetime import datetime

from utils import load_model, get_loaders, save_results

torch.multiprocessing.set_sharing_strategy('file_system')

@hydra.main(version_base=None, config_path=f"conf", config_name="config")
def main(cfg: DictConfig):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"---------")
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"---------")
    OmegaConf.set_struct(cfg, False)

    model_name = cfg.model.name
    dataset = cfg.dataset.name
    in_dim = cfg.dataset.in_dim
    config = cfg.dataset_model

    config['in_dim'] = in_dim
    
    model = load_model(model_name)
    loaders = get_loaders(dataset, config)

    wandb_logger = WandbLogger(project='DL4TSAD', name=f"{model_name}_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    METRICS = ["auc", "vus_roc", 'vus_pr']
    
    config['metrics'] = METRICS
    res_dic = {m: [] for m in METRICS}
    
    for i, (trainloader, testloader) in enumerate(loaders):
        seed_everything(0)
        print(f"Currently working on subset {i+1}/{len(loaders)}")

        config["len_loader"] = len(trainloader) # Useful for some lr scheduler
        wandb_logger.config = config

        LitModel = model(config)
        if model_name=="doc":
            LitModel.init_center(trainloader, device=DEVICE)

        precision = cfg.precision if hasattr(cfg, "precision") else None
        norm = config.max_norm if hasattr(config, "max_norm") else 0.0
        trainer = L.Trainer(
                            max_epochs=config.epochs, logger=wandb_logger, enable_checkpointing=False, 
                            log_every_n_steps=1, precision=precision, gradient_clip_val=norm 
                            )
    
        if "lr" not in config:
            config["lr"] = 1e-4  
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(
                model=LitModel, 
                train_dataloaders=trainloader
            )
            new_lr = lr_finder.suggestion()
            config["lr"] = new_lr
            print(f"Suggested learning rate: {new_lr}")

        trainer.fit(model=LitModel, train_dataloaders=trainloader)

        results = trainer.test(model=LitModel, dataloaders=testloader)
        scores = results[0]

        for k, v in scores.items():
            k = k.replace("test_", "")
            res_dic[k].append(v)
            wandb_logger.experiment.summary[f"{k}_subset_{i+1}/{len(loaders)}"] = v

        if DEVICE == "cuda": ### Free memory after each subset
            LitModel.to("cpu")
            del LitModel
            del trainloader, testloader
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            del trainer
            trainer = None
            gc.collect()
            torch.cuda.empty_cache()
            ###
        
    for k, v in res_dic.items():
        mean_v = np.mean(v)
        print(f"Final {k}: {mean_v}")
        save_results(filename=f"results/{k}.json", dataset=dataset, model=f'{model_name}{"_rev" if hasattr(config, "revin") and config.revin else ""}', score=round(mean_v, 4))
        wandb_logger.experiment.summary[f"final_{k}"] = mean_v

    wandb.finish()

if __name__ == "__main__":
    main()