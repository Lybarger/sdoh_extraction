


from collections import Counter, OrderedDict
import pandas as pd
import numpy as np
import os
import sklearn.metrics as metrics
import logging
from sklearn.metrics import classification_report

from config.constants import P, R, F1, NT, NP, FP, FN, TP, TN, SUBTYPE


METRIC_MAP = {'precision': P, "recall": R, "support": NT, "f1-score": F1}

class ScorerAnatomy:


    def __init__(self, labels):
        self.labels = labels

    def fit(self, y_true, y_pred, path=None):

        assert len(y_true) == len(y_pred), f"{len(y_true)} vs {len(y_pred)}"


        """
        Confusion matrix
        """
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, \
                                                    labels = self.labels)

        df = pd.DataFrame(confusion_matrix, \
                                                    index = self.labels,
                                                    columns = self.labels)

        confuse = []
        for row_name, columns in df.to_dict('index').items():
            for col_name, v in columns.items():
                if (row_name != col_name) and (v > 0):
                    confuse.append((row_name, col_name, v))
        df_confuse = pd.DataFrame(confuse, columns=["Gold", "Predicted", "Count"])
        df_confuse = df_confuse.sort_values("Count", ascending=False)


        if path is not None:
            f = os.path.join(path, f"confusion_matrix.csv")
            df.to_csv(f, index=False)

            f = os.path.join(path, f"confusion_scores.csv")
            df_confuse.to_csv(f, index=False)

        """
        Performance scores
        """

        scores = classification_report(y_true, y_pred, \
                                                    output_dict = True,
                                                    labels = self.labels)
        df = pd.DataFrame(scores).transpose()
        df.reset_index(inplace=True)
        df = df.rename(columns = {'index':SUBTYPE})
        df = df.rename(columns=METRIC_MAP)

        df[TP] = df[NT]*df[R]
        df[NP] = df[TP]/df[P]
        df[FP] = df[NP] - df[TP]
        df[FN] = df[NT] - df[TP]


        df = df[[SUBTYPE, NT, NP, TP, FP, FN, P, R, F1]]

        logging.info(f"Classification performance:\n{df}")

        if path is not None:
            f = os.path.join(path, f"scores.csv")
            df.to_csv(f, index=False)

        return df
