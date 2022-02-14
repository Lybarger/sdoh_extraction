

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
from scoring.scoring import PRF, summarize_event_csvs
from scipy.stats import ttest_ind

# Define experiment and load ingredients
ex = Experiment('step122_compare_scoring_criteria')


@ex.config
def cfg():


    subdir = 'train/sdoh_challenge_25_d2/'
    subdir = 'train/sdoh_review_25_d2/'

    source_dir = os.path.join(paths.extraction, subdir)

    destination = os.path.join(paths.compare_scoring_criteria,   subdir)


    score_names = [ \
                'trigExact_labeledLabel_spanExact',
                'trigOverlap_labeledLabel_spanExact',
                'trigDist_labeledLabel_spanExact',
                'trigOverlap_labeledLabel_spanOverlap',
                'trigOverlap_labeledLabel_spanPartial']

    score_files = OrderedDict([(name, f"scores_{name}.csv") for name in score_names])

    value_columns = [NT, NP, TP]
    event_types = C.EVENT_TYPES
    trigger = C.TRIGGER
    labeled_arguments = C.LABELED_ARGUMENTS
    span_only_arguments = C.SPAN_ONLY_ARGUMENTS
    argument_types = [trigger] + labeled_arguments + span_only_arguments

    scoring_dict = OrderedDict()
    scoring_dict["trigger"] = [trigger]
    scoring_dict["labeled_arguments"] = labeled_arguments
    scoring_dict["span_only_arguments"] = span_only_arguments

    # Scratch directory
    make_and_clear(destination)

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)



@ex.automain
def main(source_dir, destination, score_files, event_types, argument_types,
    trigger, labeled_arguments, span_only_arguments, scoring_dict):

    logging.info(f"Source directory: {source_dir}")

    #score_file_dict = OrderedDict()
    output = OrderedDict()

    for score_name, score_file in score_files.items():
        logging.info(f"Score file: {score_file}")
        target_file = os.path.join(source_dir, score_file)

        df = pd.read_csv(target_file)
        df.insert(0, 'score_name', score_name)

        n = len(df)
        m = 0
        for arg_name, arg_types in scoring_dict.items():

            logging.info(f"Argument type: {arg_name}")

            df_subset = df[df['argument'].isin(arg_types)]
            m += len(df_subset)

            df_totals = df_subset.sum().to_frame().transpose()
            df_totals = df_totals[[ C.NT, C.NP, C.TP]]
            df_totals = PRF(df_totals)
            df_totals.insert(0, 'score_name', score_name)

            if arg_name not in output:
                output[arg_name] = []
            output[arg_name].append(df_totals)

        if m != n:
            logging.warn(f"{score_name} - {m} vs {n}")

    for arg_name, dfs in output.items():
        df = pd.concat(dfs)
        f = os.path.join(destination, f"scoring_{arg_name}.csv")
        df.to_csv(f, index=False)

        logging.info(f"")
        logging.info(f"Argument type: {arg_name}\n{df}")

    #
    # # get all sub directories
    # result_dirs = [path for path in Path(source_dir).iterdir() if path.is_dir()]
    #
    # logging.info(f"")
    # logging.info(f"Source directory:  {source_dir}")
    # logging.info(f"Directory count:   {len(result_dirs)}")
    #
    #
    # event_rank =    {x: i for i, x in enumerate(event_types)}
    # argument_rank = {x: i for i, x in enumerate(argument_types)}
    # def ranker(row, event_rank=event_rank, argument_rank=argument_rank):
    #     event = row["event"]
    #     argument = row["argument"]
    #     y = event_rank[event] * 10 + argument_rank[argument]
    #     return y
    #

    #     dfs = []
    #     for result_dir in result_dirs:
    #
    #
    #         base = os.path.basename(result_dir)
    #
    #         target_file = os.path.join(result_dir, score_file)
    #
    #         k = (name, base)
    #         score_file_dict[k] = target_file
    #
    #         df = pd.read_csv(target_file)
    #
    #
    #
    #         df = df[df["event"].isin(event_types)]
    #         df = df[df["argument"].isin(argument_types)]
    #         df["subtype"] = df["subtype"].fillna(value='na')
    #
    #
    #         dirname = result_dir.name
    #         df["run"] = dirname
    #
    #
    #         dfs.append(df)
    #     df = pd.concat(dfs)
    #     pt = pd.pivot_table(df, values=["F1"], index=["event", "argument", "subtype"], columns=["run"])
    #
    #     pt = pt.fillna(value=0)
    #     pt.columns = pt.columns.droplevel()
    #
    #     pt = pt.reset_index()
    #
    #     pt["rank"] = pt.apply(ranker, axis=1)
    #     pt = pt.sort_values("rank")
    #     del pt["rank"]
    #
    #     f = os.path.join(destination, score_file)
    #     pt.to_csv(f)
    #
    #
    #
    # df = summarize_event_csvs(score_file_dict)
    # print(df)

    return 'Successful completion'
