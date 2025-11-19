import json

def load_model(model_name):
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
    elif model_name=="timemixer":
        from models.timemixer import TimeMixerLit as model
    elif model_name=="catch":
        from models.catch import CatchLit as model
    elif model_name=="gat":
        from models.gat import MDAT_GAT_Lit as model
    elif model_name=="moment" or model_name=="zsmoment":
        from models.moment import MomentLit as model
    elif model_name=="gpt4ts":
        from models.gpt4ts import GPT4TSLit as model
    elif model_name=="patchfm2":
        from models.patchfm import PatchFMLit as model
    else:
        assert False, f"{model_name} is not implemented"
    return model

def get_loaders(dataset, config):
    av_datasets = ["nyc_taxi", "smd", "smap", "msl", "swat", "ec2_request_latency_system_failure"]
    assert dataset in av_datasets, f"Dataset ({dataset}) should be in {av_datasets}"

    if dataset in ["ec2_request_latency_system_failure", "nyc_taxi"]:
        from dataset.nab import get_loaders as get_nab_loaders
        loaders = [get_nab_loaders(window_size=config.ws, root_dir="data/nab", dataset=dataset, batch_size=config.bs)]

    elif dataset in ["smap", "msl"]:
        from dataset.nasa import get_loaders as get_nasa_loaders, smapfiles, mslfiles
        file = smapfiles if dataset == "smap" else mslfiles
        loaders = [get_nasa_loaders(window_size=config.ws, root_dir="data/nasa", dataset=dataset, filename=f, batch_size=config.bs) for f in file]

    elif dataset == "smd":
        from dataset.smd import get_loaders as get_smd_loaders, machines
        loaders = [get_smd_loaders(window_size=config.ws, root_dir="data/smd/processed", machine=m, batch_size=config.bs) for m in machines]

    elif dataset == "swat":
        from dataset.swat import get_loaders as get_swat_loaders
        loaders = [get_swat_loaders(window_size=config.ws, root_dir="data/swat", batch_size=config.bs)]
        
    return loaders



def load_results(filename="aucs.json"):
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

def save_results(filename, dataset, model, score):
    results = load_results(filename)
    if dataset not in results:
        results[dataset]={}
    results[dataset][model] = score
    with open(filename, "w") as f:
        json.dump(results, f)


