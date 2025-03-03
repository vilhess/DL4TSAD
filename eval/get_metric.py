from sklearn.metrics import roc_auc_score, f1_score
from eval.adjust import adjust_predicts
from eval.pot import spot, dspot

def get_metrics(y_true, y_score):
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    ano_spot, _ = spot(init_set=-y_score, stream_set=-y_score, init_seuil=0.98, proba=1e-4, n_points=10)
    ano_dspot, _ = dspot(init_set=-y_score, stream_set=-y_score, init_seuil=0.98, proba=1e-4, n_points=10)

    f1_spot = f1_score(y_true=y_true, y_pred=ano_spot)
    f1_dspot = f1_score(y_true=y_true, y_pred=ano_dspot)

    adjusted_spot = adjust_predicts(y_true, ano_spot)
    adjusted_dspot = adjust_predicts(y_true, ano_dspot)

    f1_spot_adjusted = f1_score(y_true=y_true, y_pred=adjusted_spot)
    f1_dspot_adjusted = f1_score(y_true=y_true, y_pred=adjusted_dspot)

    return {
        "auc": auc,
        "f1_spot": f1_spot,
        "f1_dspot": f1_dspot,
        "f1_spot_adjusted": f1_spot_adjusted,
        "f1_dspot_adjusted": f1_dspot_adjusted,
    }