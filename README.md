# Deep Learning for Time Series Anomaly Detection (DL4TSAD)

In this repository, we implement several deep learning anomaly detection algorithms under an unsupervised framework. Our work focuses on several datasets, ranging from univariate to multivariate time series. 

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
- [**CATCH**](https://arxiv.org/pdf/2410.12261) - [GitHub: CATCH](https://github.com/decisionintelligence/CATCH)
- [**MTAD-GAT**](https://arxiv.org/pdf/2009.02040) - [GitHub: mtad-gat-pytorch](https://github.com/ML4ITS/mtad-gat-pytorch)
- [**MOMENT**](https://arxiv.org/pdf/2402.03885) - [GitHub: MOMENT](https://github.com/moment-timeseries-foundation-model/moment)
- [**TimeMixer**](https://openreview.net/pdf?id=7oLshfEIC2) - [GitHub: TimeMixer](https://github.com/kwuking/TimeMixer)
- [**PatchTrAD**](https://arxiv.org/pdf/2504.08827) - [GitHub: PatchTrAD](https://github.com/vilhess/PatchTrAD)

When available, we slightly edited the model's code from the original GitHub repository to make it work on our project. We follow the PyTorch Lightning framework.

--- 

## 📊 Datasets

### 🔹 Univariate

- **NYC Taxi Demand**  
  Location: `data/nab`

- **EC2 Request Latency (System Failure)**  
  Location: `data/nab`

Both datasets come from the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB/).

---

### 🔸 Multivariate

- **SWAT (Secure Water Treatment Testbed)**  
  Location: `data/swat`  
  Source: [iTrust, SUTD](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

- **Server Machine Dataset (SMD)**  
  Location: `data/smd`  
  Source: [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)

- **SMAP & MSL (NASA Telemetry Data)**  
  Location: `data/nasa`  
  Source:  
  [Paper](https://arxiv.org/abs/1802.04431) | [GitHub](https://github.com/khundman/telemanom)

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Datasets
To install the datasets, you can use the following command:
```bash
python dataset/preprocess.py
```
For the **SWaT** dataset, you need to claim the data from the [iTrust website](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) and place it in the `data/swat` directory. Files needed: SWaT_Dataset_Normal_v1.xlsx, SWaT_Dataset_Attack_v0.xlsx

### Training

To train a given model on a specified dataset, use the following command:

```python 
python main.py dataset=<dataset_name> model=<model_name> method=<method_name>
``` 

where `<dataset_name>`, `<model_name>` and `<method_name>` can be one of the following:  


| Models       | Datasets               | 
|-------------|------------------------|
| `aelstm`     | `nyc_taxi`            |
| `anotrans`   | `ec2_request_latency_syste` |
| `dcdetector` | `smd`                 |  
| `doc`        | `smap`                |  
| `drocc`      | `msl`                 |  
| `fedformer`  | `swat`                |  
| `lstm`       |                        | 
| `madgan`     |                        | 
| `patchad`    |                        | 
| `patchtrad`  |                        | 
| `patchtst`   |                        | 
| `tranad`     |                        | 
| `usad`       |                        | 
| `catch`       |                        |
| `MTAD-GAT`       |                        |
| `MOMENT`       |                        |
| `TimeMixer`       |                        |


---

### Testing 

During testing phase, we evaluate model's performance using solely ROC-AUC as it does not depends on setting a threshold.
Results are saved to:  
```bash
results/aucs.json
```
### Configurations

For each dataset and each model, the configurations can be view and edit in the ```conf/``` directory, following the [hydra](https://hydra.cc/) framework.
