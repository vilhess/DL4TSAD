from sklearn.metrics import roc_auc_score, f1_score
from eval.adjust import adjust_predicts
from eval.pot import spot, dspot

def get_metrics(y_true, y_score, method="spot"):
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    y_score = y_score.numpy()
    
    if method == "spot":
        detections, _ = spot(init_set=y_score, stream_set=y_score, init_seuil=0.98, proba=1e-4, n_points=10)
    elif method == "dspot":
        detections, _, _, _ = dspot(init_set=y_score, stream_set=y_score, init_seuil=0.98, proba=1e-4, n_points=10)
    else:
        raise ValueError("Method must be either spot or dspot")
    
    detections = [1 if i in detections else 0 for i in range(len(y_score))]
    f1 = f1_score(y_true=y_true, y_pred=detections)

    adjusted = adjust_predicts(y_true, detections)
    f1_adjusted = f1_score(y_true=y_true, y_pred=adjusted)

    return {
        "auc": auc,
        "f1": f1,
        "f1_adjusted": f1_adjusted,
    }