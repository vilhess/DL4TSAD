import torch
import numpy as np
from lightning.pytorch import seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf
import gc

from model import PatchEncoder
from load_data import load_all_sets
from dataset import PaAnoSignalProcessor
from trainer import PAanoTrainer

import sys 
sys.path.append("../../")
from utils import save_results

torch.multiprocessing.set_sharing_strategy('file_system')

@hydra.main(version_base=None, config_path=f"../../conf", config_name="config")
def main(cfg: DictConfig):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"---------")
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"---------")
    OmegaConf.set_struct(cfg, False)

    model_name = cfg.model.name
    assert model_name=="paano", "This runner is only for PaAno, please change the model name in the config to 'paano'"

    dataset = cfg.dataset.name
    in_dim = cfg.dataset.in_dim
    config = cfg.dataset_model

    config['in_dim'] = in_dim
    
    all_sets = load_all_sets(dataset, window_size=config['patch_size'])

    METRICS = ["auc", "vus_roc", 'vus_pr']
    
    config['metrics'] = METRICS
    res_dic = {m: [] for m in METRICS}
    
    for i, (train_data, test_data, test_labels) in enumerate(all_sets):
        seed_everything(0)
        print(f"Currently working on subset {i+1}/{len(all_sets)}")

        model = PatchEncoder(config)
        preprocessor = PaAnoSignalProcessor(train_data, test_data, test_labels, patch_size=config['patch_size'], stride=config['stride'])
        trainloader, testloader, test_labels = preprocessor.get_loaders(batch_size=config['bs'])
        train_patches = preprocessor.get_all_patches(set="train")

        trainer = PAanoTrainer(model, config, DEVICE)
        trainer.fit(trainloader, train_patches)

        scores = trainer.test(testloader, test_labels)

        for k, v in scores.items():
            res_dic[k].append(v)
            print(f"Subset {i+1}/{len(all_sets)} - {k}: {v}")

        if DEVICE == "cuda": ### Free memory after each subset
            model.to("cpu")
            del model
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
        save_results(filename=f"../../results/{k}.json", dataset=dataset, model=f'{model_name}', score=round(mean_v, 4))

if __name__ == "__main__":
    main()