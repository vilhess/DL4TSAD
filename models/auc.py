import torch
from sklearn.metrics import roc_auc_score

class StreamAUC:
    def __init__(self):
        self.test_scores = []
        self.test_labels = []
    
    def update(self, errors, labels):

        self.test_scores.append(errors)
        self.test_labels.append(labels)
    
    def compute(self):
        self.test_scores = torch.cat(self.test_scores).detach().cpu().numpy()
        self.test_labels = torch.cat(self.test_labels).detach().cpu().numpy()

        auc = roc_auc_score(y_true=self.test_labels, y_score=self.test_scores)
        return auc
    
    def reset(self):
        self.test_scores = []
        self.test_labels = []
        return