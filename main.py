import torch
import numpy as np
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import gc

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
    config = cfg.dataset_model
    
    model = load_model(model_name)
    loaders = get_loaders(dataset, config)

    wandb_logger = WandbLogger(project='DL4TSAD', name=f"{model_name}_{dataset}")
    aucs = []
    
    for i, (trainloader, testloader) in enumerate(loaders):
        torch.manual_seed(0)
        print(f"Currently working on subset {i+1}/{len(loaders)}")

        config["len_loader"] = len(trainloader) # Useful for some lr scheduler
        wandb_logger.config = config

        LitModel = model(config)
        if model_name=="doc":
            LitModel.init_center(trainloader, device=DEVICE)

        precision = cfg.precision if hasattr(cfg, "precision") else None
        trainer = L.Trainer(max_epochs=config.epochs, logger=wandb_logger, enable_checkpointing=False, log_every_n_steps=1, precision=precision)

        trainer.fit(model=LitModel, train_dataloaders=trainloader)

        results = trainer.test(model=LitModel, dataloaders=testloader)
        auc = results[0]["auc"]
        print(f"AUC: {auc}")

        aucs.append(auc)
        wandb_logger.experiment.summary[f"auc_subset_{i+1}/{len(loaders)}"] = auc


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
        
    final_auc = np.mean(aucs)
    print(f"Final AUC: {final_auc}")
    save_results(filename="results/aucs.json", dataset=dataset, model=f'{model_name}{"_rev" if hasattr(config, "revin") and config.revin else ""}', score=round(final_auc, 4))
    wandb_logger.experiment.summary[f"final_auc"] = final_auc

    wandb.finish()

if __name__ == "__main__":
    main()