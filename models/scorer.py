import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from vus.metrics import get_metrics
from vus.utils.utility import get_list_anomaly

class StreamScorer:
    def __init__(self, metrics=['auc', "vus_pr", "vus_roc"]):

        if type(metrics) is str: metrics = [metrics]
        for metric in metrics:
            if metric not in ['auc', 'vus_roc', 'vus_pr']:
                print(f"Metric {metric} not implemented for StreamScorer, removing it.")
                metrics.remove(metric)
        if len(metrics) == 0:
            raise ValueError("No valid metrics provided for StreamScorer.")

        self.test_scores = []
        self.test_labels = []
        self.metrics = metrics
    
    def update(self, errors, labels):

        self.test_scores.append(errors)
        self.test_labels.append(labels)
    
    def compute(self):
        self.test_scores = torch.cat(self.test_scores).detach().cpu().numpy()
        self.test_labels = torch.cat(self.test_labels).detach().cpu().numpy()

        results = {}

        if "auc" in self.metrics:
            auc = roc_auc_score(y_true=self.test_labels, y_score=self.test_scores)
            results['auc'] = auc

        if "vus_pr" in self.metrics or "vus_roc" in self.metrics:
            slidingWindow = max(int(get_sliding_window(self.test_labels)), 10)
            metrics = get_metrics(self.test_scores, self.test_labels, slidingWindow=slidingWindow, metric="vus")
            results["vus_roc"] = metrics["VUS_ROC"]
            results["vus_pr"] = metrics["VUS_PR"]

        return results
    
    def reset(self):
        self.test_scores = []
        self.test_labels = []
        return
    
def get_sliding_window(labels):
    return np.median(get_list_anomaly(labels))