

import numpy as np
import pandas as pd
import os
import json
from collections import Counter, OrderedDict
from sklearn.metrics import classification_report, precision_recall_fscore_support
import logging


def get_sent_labels(X):

    Y = OrderedDict()
    for x in X:

        sent_labels = x["sent_labels"]

        for k, v in sent_labels.items():
            if k not in Y:
                Y[k] = []
            Y[k].append(v)

    return Y

def score_sent_labels(gold, predict, destination):

    rows = []
    for k in gold:

        g = gold[k]
        p = predict[k]

        assert set(g).issubset(set([0, 1]))
        assert set(p).issubset(set([0, 1]))

        P, R, F1, support = precision_recall_fscore_support(g, p, \
                                                    average = 'binary',
                                                    pos_label = 1,
                                                    zero_division = 0)
        NT = sum(g)
        NP = sum(p)
        TP = int(R*NT)

        rows.append((k, NT, NP, TP, P, R, F1))

    df_detail = pd.DataFrame(rows, columns=['type', 'NT', 'NP', 'TP', 'P', 'R', 'F1'])

    f = os.path.join(destination, "scores_sent_labels_detail.csv")
    df_detail.to_csv(f, index=False)

    return df_detail

def score_spert_docs(gold_file, predict_file, destination):

    gold = json.load(open(gold_file, "r"))
    predict = json.load(open(predict_file, "r"))


    assert len(gold) == len(predict)


    if "sent_labels" in gold[0]:

        gold_sent_labels = get_sent_labels(gold)
        predict_sent_labels = get_sent_labels(predict)
        df = score_sent_labels(gold_sent_labels, predict_sent_labels, destination)



    return True


#
# gold_file = "/home/lybarger/incidentalomas/analyses/step102_anatomy_extraction/train/unknown2_FAST_RUN/data_valid.json"
# predict_file = "/home/lybarger/incidentalomas/analyses/step102_anatomy_extraction/train/unknown2_FAST_RUN/log/predictions.json"
# destination = "/home/lybarger/incidentalomas/analyses/step102_anatomy_extraction/train/unknown2_FAST_RUN/"
# score_spert_docs(gold_file, predict_file, destination)
