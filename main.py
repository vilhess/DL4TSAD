import torch
import numpy as np
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataset.nab import get_loaders as get_nab_loaders
from dataset.nasa import get_loaders as get_nasa_loaders, smapfiles, mslfiles
from dataset.smd import get_loaders as get_smd_loaders, machines
from dataset.swat import get_loaders as get_swat_loaders

from eval.get_metric import get_metrics
from utils import save_results

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
    method = cfg.method.name
    threshold = cfg.dataset.threshold
    
    if model_name=="patchtrad":
        from models.patchtrad import PatchTradLit as model
    elif model_name=="aelstm":
        from models.aelstm import AELSTMLit as model
    elif model_name=="tranad":
        from models.tranad import TranADLit as model
    elif model_name=="lstm":
        from models.lstm import LSTMLit as model
    elif model_name=="fedformer":
        from models.fedformer import FEDformerLit as model
    elif model_name=="patchad":
        from models.patchad import PatchADLit as model
    elif model_name=="doc":
        from models.doc import DOCLit as model
    elif model_name=="anotrans":
        from models.anotrans import AnomalyTransformerLit as model
    elif model_name=="dcdetector":
        from models.dcdetector import DCDetectorLit as model
    elif model_name=="drocc":
        from models.drocc import DROCCLit as model
    elif model_name=="patchtst":
        from models.patchtst import PatchTSTLit as model
    elif model_name=="usad":
        from models.usad import USADLit as model
    elif model_name=="madgan":
        from models.madgan import MADGANLit as model

    av_datasets = ["nyc_taxi", "smd", "smap", "msl", "swat", "ec2_request_latency_system_failure"]
    assert dataset in av_datasets, f"Dataset ({dataset}) should be in {av_datasets}"

    if dataset in ["ec2_request_latency_system_failure", "nyc_taxi"]:
        loaders = [get_nab_loaders(window_size=config.ws, root_dir="data/nab", dataset=dataset, batch_size=config.bs)]
    elif dataset in ["smap", "msl"]:
        file = smapfiles if dataset == "smap" else mslfiles
        loaders = [get_nasa_loaders(window_size=config.ws, root_dir="data/nasa", dataset=dataset, filename=f, batch_size=config.bs) for f in file]
    elif dataset == "smd":
        loaders = [get_smd_loaders(window_size=config.ws, root_dir="data/smd/processed", machine=m, batch_size=config.bs) for m in machines]
    elif dataset == "swat":
        loaders = [get_swat_loaders(window_size=config.ws, root_dir="data/swat", batch_size=config.bs)]

    wandb_logger = WandbLogger(project='DL4TSAD', name=f"{model_name}_{dataset}")

    aucs, f1, f1_adjusted = [], [], []
    
    for i, (trainloader, testloader) in enumerate(loaders):
        torch.manual_seed(0)
        print(f"Currently working on subset {i+1}/{len(loaders)}")

        config["len_loader"] = len(trainloader) # Useful for some lr scheduler

        LitModel = model(config)
        if model_name=="doc":
            LitModel.init_center(trainloader)
        trainer = L.Trainer(max_epochs=config.epochs, logger=wandb_logger, enable_checkpointing=False, log_every_n_steps=1)
        #trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False, fast_dev_run=True)
        trainer.fit(model=LitModel, train_dataloaders=trainloader)
        
        test_errors = []
        test_labels = []

        LitModel = LitModel.to(DEVICE)
        LitModel.eval()

        with torch.no_grad():
            pbar = tqdm(testloader, desc="Detection Phase")
            for x, anomaly in pbar:
                x = x.to(DEVICE)
                errors = LitModel.get_loss(x, mode="test")

                test_labels.append(anomaly)
                test_errors.append(errors)

        test_errors = torch.cat(test_errors).detach().cpu()
        test_labels = torch.cat(test_labels).detach().cpu()

        results = get_metrics(test_labels, test_errors, return_f1=False, method=method, threshold=threshold)
        print(f"Results: {results}")

        aucs.append(results["auc"])
        #f1.append(results["f1"])
        #f1_adjusted.append(results["f1_adjusted"])

        wandb_logger.experiment.config[f"auc_subset_{i+1}/{len(loaders)}"] = results["auc"]
        #wandb_logger.experiment.config[f"f1_subset_{i+1}/{len(loaders)}_{method}"] = results["f1"]
        #wandb_logger.experiment.config[f"f1_adjusted_subset_{i+1}/{len(loaders)}_{method}"] = results["f1_adjusted"]
    
    final_auc = np.mean(aucs)
    #final_f1 = np.mean(f1)
    #final_adjusted = np.mean(f1_adjusted)

    print(f"Final AUC: {final_auc}")
    #print(f"Final F1: {final_f1}")
    #print(f"Final F1-Adjusted: {final_adjusted}")

    save_results(filename="results/aucs.json", dataset=dataset, model=f'{model_name}{"_rev" if hasattr(config, "revin") and config.revin else ""}', score=round(final_auc, 4))
    #save_results(filename="results/f1.json", dataset=dataset, model=f"{model_name}{"_rev" if hasattr(config, "revin") and config.revin else ""}_{method}", score=round(final_f1, 4))
    #save_results(filename="results/f1_adjusted.json", dataset=dataset, model=f"{model_name}{"_rev" if hasattr(config, "revin") and config.revin else ""}_{method}", score=round(final_adjusted, 4))

    wandb_logger.experiment.config["final_auc"] = final_auc
    #wandb_logger.experiment.config[f"final_f1_{method}"] = final_f1
    #wandb_logger.experiment.config[f"final_f1_adjusted_{method}"] = final_adjusted
    wandb.finish()

if __name__ == "__main__":
    main()