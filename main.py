import torch
from sklearn.metrics import roc_auc_score
import numpy as np
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataset.nab import get_loaders as get_nab_loaders
from dataset.nasa import get_loaders as get_nasa_loaders, smapfiles, mslfiles
from dataset.smd import get_loaders as get_smd_loaders, machines
from dataset.swat import get_loaders as get_swat_loaders

from utils import save_results


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
    elif model_name=="fdad":
        from models.fdad import FDADLit as model

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
    
    aucs = []
    
    for i, (trainloader, testloader) in enumerate(loaders):
        torch.manual_seed(0)
        print(f"Currently working on subset {i+1}/{len(loaders)}")

        config["len_loader"] = len(trainloader) # Useful for some lr scheduler

        LitModel = model(config)
        if model_name=="doc":
            LitModel.init_center(trainloader)
        trainer = L.Trainer(max_epochs=config.epochs, logger=False, enable_checkpointing=False)
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

        test_scores = -test_errors
        
        auc = roc_auc_score(y_true=test_labels, y_score=test_scores)
        print(f"AUC: {auc}")
        aucs.append(auc)
    
    final_auc = np.mean(aucs)
    print(f"Final AUC: {final_auc}")
    save_results(filename="results/results.json", dataset=dataset, model=model_name, auc=round(final_auc, 4))

if __name__ == "__main__":
    main()
