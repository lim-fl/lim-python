from collections import namedtuple
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

_results = []

Scores = namedtuple("Scores",
                    [
                        "classifier",
                        "federation_round",
                        "place",
                        "id",
                        "tn",
                        "fp",
                        "fn",
                        "tp",
                        "f1",
                        "accuracy",
                        "poisoned",
                        "weights",
                        "membership_tp",
                        "membership_fp",
                    ])


def add(
        place: str,
        federation_round: int,
        id_: int,
        labels: dict,
        predictions: dict,
        poisoned: bool,
        weights: dict,
        membership_tp=0,
        membership_fp=0):

    for classifier in predictions:
        y_pred = predictions[classifier]
        y_true = labels[classifier]
        if classifier in weights:
            w = weights[classifier]
        else:
            w = None
        if y_pred is not None and y_true is not None:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
            if tp + fn > 0:
                f1 = f1_score(y_true, y_pred)
            else:
                f1 = np.nan
            accuracy = accuracy_score(y_true, y_pred)
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            f1 = 0
            accuracy = 0

        scores = Scores(
            classifier=classifier,
            federation_round=federation_round,
            id=id_,
            place=place,
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            f1=f1,
            poisoned=poisoned,
            accuracy=accuracy,
            weights=w,
            membership_tp=membership_tp,
            membership_fp=membership_fp,
        )

        _results.append(scores)


def to_df():
    return pd.DataFrame(data=_results)


def clear():
    _results.clear()
