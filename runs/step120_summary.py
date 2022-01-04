

from __future__ import division, print_function, unicode_literals

from sacred import Experiment
from sacred.observers import FileStorageObserver

from pathlib import Path
import os
import re
import numpy as np
import json
import joblib

from tqdm import tqdm
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#pd.set_option('display.max_rows', None)
from collections import Counter, OrderedDict
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.custom_observer import CustomObserver
from utils.proj_setup import make_and_clear

from config.constants import CV, FIT, PREDICT, SCORE, PROB, SUBSET, TRAIN, TEST, DEV, CORPUS_FILE, PARAMS_FILE, TYPE, SUBTYPE
from config.constants import ENTITIES, SCORES_FILE, NT, NP, TP, P, R, F1, SUBTYPE
import config.constants as C
import config.paths as paths
from corpus.tokenization import get_tokenizer, get_context
from scoring.scoring import PRF
from scipy.stats import ttest_ind

# Define experiment and load ingredients
ex = Experiment('step120_summary')


@ex.config
def cfg():

    source = "sdoh"

    dirname = 'train'

    source_dir = os.path.join(paths.extraction, dirname)
    destination = os.path.join(paths.summary,   dirname)

    score_files = ["scores_relaxed.csv", "scores_strict.csv"]

    value_columns = [NT, NP, TP]
    event_types = C.EVENT_TYPES
    argument_types = [C.TRIGGER] + C.LABELED_ARGUMENTS + C.SPAN_ONLY_ARGUMENTS


    #params = ["epochs", "prop_drop", "lr", "subtype_classification"]

    #target_run = "e10_lr55_d02_bs10_ss1_ctL_ps___pd__"



    # Scratch directory
    make_and_clear(destination)

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)


# def get_config(dir, file_name="config.json"):
#
#     target = None
#     max_dir = -1
#     for path in Path(dir).iterdir():
#         if path.is_dir() and path.name.isnumeric() and \
#             (int(path.name) > max_dir):
#
#             target = path
#             max_dir = int(path.name)
#
#     assert target is not None
#
#     f = os.path.join(target, file_name)
#     config = json.load(open(f, "r"))
#
#     return config



@ex.automain
def main(source_dir, destination, score_files, event_types, argument_types):


    # get all sub directories
    result_dirs = [path for path in Path(source_dir).iterdir() if path.is_dir()]

    logging.info(f"")
    logging.info(f"Source directory:  {source_dir}")
    logging.info(f"Directory count:   {len(result_dirs)}")


    event_rank =    {x: i for i, x in enumerate(event_types)}
    argument_rank = {x: i for i, x in enumerate(argument_types)}
    def ranker(row, event_rank=event_rank, argument_rank=argument_rank):
        event = row["event"]
        argument = row["argument"]
        y = event_rank[event] * 10 + argument_rank[argument]
        return y

    for score_file in score_files:
        dfs = []
        for result_dir in result_dirs:
            target_file = os.path.join(result_dir, score_file)
            df = pd.read_csv(target_file)

            df = df[df["event"].isin(event_types)]
            df = df[df["argument"].isin(argument_types)]
            df["subtype"] = df["subtype"].fillna(value='na')


            dirname = result_dir.name
            df["run"] = dirname
            

            dfs.append(df)
        df = pd.concat(dfs)
        pt = pd.pivot_table(df, values=["F1"], index=["event", "argument", "subtype"], columns=["run"])

        pt = pt.fillna(value=0)
        pt.columns = pt.columns.droplevel()

        pt = pt.reset_index()

        pt["rank"] = pt.apply(ranker, axis=1)
        pt = pt.sort_values("rank")
        del pt["rank"]
        print(pt)
        f = os.path.join(destination, score_file)
        pt.to_csv(f)





    return 'Successful completion'
