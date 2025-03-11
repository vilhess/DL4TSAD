# Deep Learning for Time Series Anomaly Detection (DL4TSAD)

In this repository, we implement several deep learning anomaly detection algorithms under an unsupervised framework. Our work focuses on several datasets, ranging from univariate to multivariate time series. 

---

## Datasets

### Univariate

> **NYC Taxi Demand Dataset** The dataset should be placed in the root directory under `data/nab`.    
> **EC2 Request Latency System Failure Dataset** The dataset should be placed in the root directory under `data/nab`.    

- These two datasets are sourced from the Numenta Anomaly Benchmark (NAB) and can be accessed [here](https://github.com/numenta/NAB/).

### Multivariate

> **SWAT Dataset**: Place the files in the root directory under `data/swat`.

- The dataset is provided by iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design. More details and access requests can be made [here](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/).

- ⚠ **Important:** Preprocessing is required before using the data.  
- The necessary preprocessing functions can be found in `dataset/swat`.



> **Server Machine Dataset** The files should be placed in the root directory under `data/smd`.  

- This dataset originates from the **OmniAnomaly** methods and can be downloaded by cloning the original [OmniAnomaly repository](https://github.com/NetManAIOps/OmniAnomaly).

- ⚠ **Important:** Preprocessing is required before using the data.  
- The necessary preprocessing functions can be found in `dataset/smd`.



> **SMAP Dataset** The files should be placed in the root directory under `data/nasa`.  
> **MSL Dataset** The files should be placed in the root directory under `data/nasa`.  

- These datasets contains expert-labeled telemetry anomaly data from the Soil Moisture Active Passive (SMAP) satellite and the Mars Science Laboratory (MSL). They are provided from the **NASA Jet Propulsion Laboratory** [Paper](https://arxiv.org/abs/1802.04431) [GitHub](https://github.com/khundman/telemanom)

> To process directly the **Server Machine Dataset** and the **SWAT Dataset**, you can use the following command:
```python
python dataset/preprocess.py
```

---


## Models

The models we consider are:
- **LSTM predictive model** 
- [**USAD**](https://dl.acm.org/doi/10.1145/3394486.3403392)
- [**Deep One-Class Classification**](http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf) - [GitHub: Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch)
- **LSTM Auto-Encoders** 
- [**DROCC**](https://arxiv.org/abs/2002.12718) - [GitHub: EdgeML](https://github.com/microsoft/EdgeML/tree/master)
- [**TranAD**](https://arxiv.org/abs/2201.07284) - [GitHub: TranAD](https://github.com/imperial-qore/TranAD)
- [**FEDformer**](https://arxiv.org/abs/2201.12740) - [GitHub: FEDformer](https://github.com/MAZiqing/FEDformer)
- [**PatchTST**](https://arxiv.org/abs/2211.14730) - [GitHub: PatchTST](https://github.com/yuqinie98/PatchTST)
- [**Revin Method**](https://openreview.net/forum?id=cGDAkQo1C0p) - [GitHub: RevIN](https://github.com/ts-kim/RevIN)
- [**Anomaly Transformer**](https://arxiv.org/abs/2110.02642) - [GitHub: Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer)
- [**DCDetector**](https://arxiv.org/abs/2306.10347) - [GitHub: KDD2023-DCdetector](https://github.com/DAMO-DI-ML/KDD2023-DCdetector)
- [**PatchAD**](https://arxiv.org/abs/2401.09793) - [GitHub: PatchAD](https://github.com/EmorZz1G/PatchAD)
- [**USAD**](https://dl.acm.org/doi/10.1145/3394486.3403392) 
- [**MADGAN**](https://arxiv.org/abs/1901.04997) - [GitHub: madgan-pytorch](https://github.com/Guillem96/madgan-pytorch)

We slightly edited the model's code from the original GitHub repository to make it work on our project. We follow the PyTorch Lightning framework.

--- 

## Training

To train a given model on a specified dataset, use the following command:

```python 
python main.py dataset=<dataset_name> model=<model_name> method=<method_name>
``` 

where `<dataset_name>`, `<model_name>` and `<method_name>` can be one of the following:  


| Models       | Datasets               | Method  
|-------------|------------------------|----------
| `aelstm`     | `nyc_taxi`            |`spot`  |
| `anotrans`   | `ec2_request_latency_system_failure` |`dspot`|  
| `dcdetector` | `smd`                 |        |
| `doc`        | `smap`                |        |
| `drocc`      | `msl`                 |        |
| `fedformer`  | `swat`                |        |
| `lstm`       |                        |        |
| `madgan`     |                        |        |
| `patchad`    |                        |        |
| `patchtrad`  |                        |        |
| `patchtst`   |                        |        |
| `tranad`     |                        |        |
| `usad`       |                        |        |


---

## Testing 

During the testing phase, we evaluate the model's performance using ROC-AUC and F1 scores. For anomaly detection, we employ adaptive thresholding based on techniques from [Anomaly Detection in Streams with Extreme Value Theory](https://hal.science/hal-01640325/document). Both `SPOT` and `DSPOT` methods are available.

## Configurations

For each dataset and each model, the configurations can be view and edit in the ```conf/``` directory, following the [hydra](https://hydra.cc/) framework.
