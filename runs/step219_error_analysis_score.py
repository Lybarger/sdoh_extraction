
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.insert(0, '/home/lybarger/brat_scoring')

import argparse
from zipfile import ZipFile
from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
import shutil
import logging
import json
import config.paths as paths
import config.constants as C
from utils.path_utils import create_project_folder, define_logging
import re
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import glob
pd.set_option('display.max_rows', None)
import joblib
from pathlib import Path

from brat_scoring.corpus import Corpus
from brat_scoring.scoring import score_docs
from brat_scoring.constants import EXACT, LABEL, OVERLAP, PARTIAL, MIN_DIST, SPACY_MODEL, EVENT, OVERALL, F1, P, R, NT, NP, TP, ARGUMENT, TRIGGER, SUBTYPE
from brat_scoring.constants_sdoh import STATUS_TIME, TYPE_LIVING, STATUS_EMPLOY
from brat_scoring.constants_sdoh import LABELED_ARGUMENTS, NONE, CURRENT, PAST

import brat_scoring.constants_sdoh as SC




EVENT_ARG_PAIRS = { \
    SC.ALCOHOL:        SC.STATUS_TIME,
    SC.DRUG:           SC.STATUS_TIME,
    SC.TOBACCO:        SC.STATUS_TIME,
    SC.EMPLOYMENT:     SC.STATUS_EMPLOY,
    SC.LIVING_STATUS:  SC.TYPE_LIVING
    }

EVENT_TYPES = [SC.ALCOHOL, SC.DRUG, SC.TOBACCO, SC.EMPLOYMENT, SC.LIVING_STATUS]
SUBSTANCE_TYPES = [SC.ALCOHOL, SC.DRUG, SC.TOBACCO]

UNKOWN = 'unknown'
SIMPLE = 'simple'
COMPLEX = 'complex'
CATEGORY = 'Category'

ID = 'ID'
SUBSET = "subset"
SOURCE = "source"
NUM = 'num'
SUBTASK = "subtask"
SUBMISSION = 'submission'

SOURCE_SUBSET = f'{SOURCE}_{SUBSET}'

# def import_submission(ann_path, txt_path):

#     corpus = Corpus()
#     summary, detailed = corpus.import_predictions( \
#                     txt_path = txt_path,
#                     ann_path = ann_path,
#                     strict_import = False)

#     return (corpus, summary, detailed)

def evt_count_pat(event_type):
    return f'{event_type}_Count'


def sub_pat(event_type):
    return f'{event_type}_None'


def score_brat(path, gold_corpus, predict_corpus, \
            labeled_args, score_trig, score_span, score_labeled,
                            description = None,
                            include_detailed = False,
                            spacy_model = SPACY_MODEL,
                            event_types = None,
                            argument_types = None,
                            param_dict = None):


    assert gold_corpus.doc_count()    > 0, f"Could not find any BRAT files at: {gold_dir}"
    assert predict_corpus.doc_count() > 0, f"Could not find any BRAT files at: {predict_dir}"

    gold_docs =    gold_corpus.docs(as_dict=True)
    predict_docs = predict_corpus.docs(as_dict=True)



    df_corpus = score_docs(gold_docs, predict_docs, \
                                labeled_args = labeled_args,
                                score_trig = score_trig,
                                score_span = score_span,
                                score_labeled = score_labeled,
                                output_path = None,
                                description = description,
                                include_detailed = include_detailed,
                                spacy_model = spacy_model,
                                event_types = event_types,
                                argument_types = argument_types,
                                param_dict = param_dict,
                                verbose = False)

    f = os.path.join(path, "scores_corpus.csv")
    df_corpus.to_csv(f)

    doc_scores = []
    for k in gold_docs.keys():
        g = {k: gold_docs[k]}
        p = {k: predict_docs[k]}

        df = score_docs(g, p, \
                                labeled_args = labeled_args,
                                score_trig = score_trig,
                                score_span = score_span,
                                score_labeled = score_labeled,
                                output_path = None,
                                description = description,
                                include_detailed = include_detailed,
                                spacy_model = spacy_model,
                                event_types = event_types,
                                argument_types = argument_types,
                                param_dict = param_dict,
                                verbose = False)

        df = df[~(df[EVENT] == OVERALL)]

        subset, source, num = k.split("/")

        df.insert(0, ID, k)
        df.insert(1, SUBSET, subset)
        df.insert(2, SOURCE, source)
        df.insert(3, NUM, num)
        df.insert(4, SOURCE_SUBSET, f'{source}_{subset}')
            
        for evt_typ, _ in EVENT_ARG_PAIRS.items():
            df_temp = df[ \
                    (df[EVENT]    == evt_typ) &
                    (df[ARGUMENT] == TRIGGER)
                    ]
            df[evt_count_pat(evt_typ)] = df_temp[NT].sum()

        '''
        Substance use
        '''
        for evt_typ in SUBSTANCE_TYPES:
            df_temp = df[ \
                    (df[EVENT]    == evt_typ) &
                    (df[ARGUMENT] == STATUS_TIME)
                    ]

            count_all = df_temp[NT].sum()

            df_none = df_temp[df_temp[SUBTYPE]  == NONE]
            count_none = df_none[NT].sum()

            df_pos = df_temp[df_temp[SUBTYPE].isin([CURRENT, PAST])]
            count_pos = df_pos[NT].sum()

            if count_all != count_none + count_pos:
                logging.warn(f'Substance status count mismatch: {count_all} vs {count_none + count_pos}\n{df_temp}')

            if count_all == 0:
                df[f'{evt_typ}_{CATEGORY}'] = UNKOWN

            elif (count_none > 0) and (count_pos == 0):
                df[f'{evt_typ}_{CATEGORY}'] = NONE

            elif (count_pos == 1):
                df[f'{evt_typ}_{CATEGORY}'] = SIMPLE

            elif (count_pos > 1):
                df[f'{evt_typ}_{CATEGORY}'] = COMPLEX
            else:
                raise ValueError("Could not define substance category")

        '''
        Employment
        '''

        df_temp = df[ \
                (df[EVENT]    == SC.EMPLOYMENT) &
                (df[ARGUMENT] == SC.STATUS_EMPLOY) &
                (df[NT]       >  0)
                ]

        num_subtypes = len(df_temp)

        if (num_subtypes == 0):
            df[f'{SC.EMPLOYMENT}_{CATEGORY}'] = UNKOWN

        elif (num_subtypes == 1):
            subtype = df_temp[SUBTYPE].tolist()[0]
            df[f'{SC.EMPLOYMENT}_{CATEGORY}'] = subtype

        elif (num_subtypes > 1):
            df[f'{SC.EMPLOYMENT}_{CATEGORY}'] = COMPLEX
        else:
            raise ValueError("Could not define employment category")

        '''
        Living Status
        '''

        df_temp = df[ \
                (df[EVENT]    == SC.LIVING_STATUS) &
                (df[ARGUMENT] == SC.TYPE_LIVING) &
                (df[NT]       >  0)
                ]

        num_subtypes = len(df_temp)

        if (num_subtypes == 0):
            df[f'{SC.LIVING_STATUS}_{CATEGORY}'] = UNKOWN

        elif (num_subtypes == 1):
            subtype = df_temp[SUBTYPE].tolist()[0]
            df[f'{SC.LIVING_STATUS}_{CATEGORY}'] = subtype

        elif (num_subtypes > 1):
            df[f'{SC.LIVING_STATUS}_{CATEGORY}'] = COMPLEX
        else:
            raise ValueError("Could not define employment category")

        doc_scores.append(df)

    df = pd.concat(doc_scores)
    df.index = list(range(len(df)))

    f = os.path.join(path, "scores_docs.csv")
    df.to_csv(f)

    combos = df[SOURCE_SUBSET].unique()
    rows = []
    for c in combos:
        df_temp = df[df[SOURCE_SUBSET] == c]

        f = os.path.join(path, f"scores_docs_{c}_all.csv")
        df_temp.to_csv(f)

        d =  {}
        d[SOURCE_SUBSET] = c
        d.update(get_prf(df_temp))
        rows.append(d)

    df_src_sub = pd.DataFrame(rows)
    f = os.path.join(path, "scores_source_subset.csv")
    df_src_sub.to_csv(f)


    return (df, df_corpus)


def score_subtask(subtask, destination_dir, gold_file, predict_file, \
                            labeled_args = LABELED_ARGUMENTS,
                            score_trig = OVERLAP,
                            score_span = EXACT,
                            score_labeled = LABEL):

    logging.info(f'-'*80)
    logging.info(f'score_subtask')
    logging.info(f'-'*80)
    logging.info('')
    logging.info(f'INPUTS:')
    logging.info(f'subtask:             {subtask}')
    logging.info(f'destination_dir:     {destination_dir}')
    logging.info(f'gold_files:          {gold_file}')
    logging.info(f'predict_file:        {predict_file}')
    logging.info(f'labeled_args:        {labeled_args}')
    logging.info(f'score_trig:          {score_trig}')
    logging.info(f'score_span:          {score_span}')
    logging.info(f'score_labeled:       {score_labeled}')
    logging.info('')

    # load corpora
    gold_corpus = joblib.load(gold_file)
    predict_corpus = joblib.load(predict_file)

    # score corpora
    df_docs, df_corpus = score_brat( \
            path = destination_dir,
            gold_corpus = gold_corpus,
            predict_corpus = predict_corpus,
            labeled_args = labeled_args,
            score_trig = score_trig,
            score_span = score_span,
            score_labeled = score_labeled,
            include_detailed = False)        


    df_corpus = df_corpus[(df_corpus[EVENT] == OVERALL)]


    f = os.path.join(destination_dir, f"scores_docs.csv")
    df_docs.to_csv(f)

    f = os.path.join(destination_dir, f"scores_corpus.csv")
    df_corpus.to_csv(f)

    return df_docs


def score_submissions(destination_dir, gold_file, predict_file, \
                            labeled_args = LABELED_ARGUMENTS,
                            score_trig = OVERLAP,
                            score_span = EXACT,
                            score_labeled = LABEL,
                            fast_run = False):



    logging.info(f'-'*80)
    logging.info(f'score_submissions')
    logging.info(f'-'*80)
    logging.info('')
    logging.info(f'INPUTS:')
    logging.info(f'destination_dir:    {destination_dir}')
    logging.info(f'gold_file:          {gold_file}')
    logging.info(f'predict_file:       {predict_file}')
    logging.info(f'labeled_args:       {labeled_args}')
    logging.info(f'score_trig:         {score_trig}')
    logging.info(f'score_span:         {score_span}')
    logging.info(f'score_labeled:      {score_labeled}')
    logging.info(f'fast_run:           {fast_run}')
    logging.info('')



    df = score_subtask( \
            subtask = 'none',
            destination_dir = destination_dir,
            gold_file = gold_file,
            predict_file = predict_file,
            labeled_args = labeled_args,
            score_trig = score_trig,
            score_span = score_span,
            score_labeled = score_labeled)

    return df


def get_prf(df, name=None):

    nt_ = df["NT"].sum()
    np_ = df["NP"].sum()
    tp_ = df["TP"].sum()

    # precision
    if np_ == 0:
        p_ = 0
    else:
        p_ = tp_/np_

    # recall
    if nt_ == 0:
        r_ = 0
    else:
        r_ = tp_/nt_

    # f1
    if p_ + r_ == 0:
        f1_ = 0
    else:
        f1_ = 2*p_*r_/(p_ + r_)

    d = {}
    if name is not None:
        d['name'] = name

    d.update(dict(NT=nt_, NP=np_, TP=tp_, P=p_, R=r_, F1=f1_))

    return d

def main(args):



    destination_dir = paths.error_analysis_score

    source_dir = paths.error_analysis_import
    predict_file = os.path.join(source_dir, "predict.pkl")
    gold_file = os.path.join(source_dir, "gold.pkl")

    create_project_folder(destination_dir)

    define_logging(destination_dir)


    logging.info("")
    logging.info('='*80)
    logging.info('step219_error_analysis_score')
    logging.info('='*80)


    logging.info(f'destination_dir:     {destination_dir}')
    logging.info(f'source_dir:          {source_dir}')
    logging.info(f'predict_file:        {predict_file}')
    logging.info(f'gold_file:           {gold_file}')


    df = score_submissions( \
                destination_dir = destination_dir,
                gold_file = gold_file,
                predict_file = predict_file)

    f = os.path.join(destination_dir, 'scores.csv')
    df.to_csv(f)

    # df_scores, df_delta_test, df_p_values = sig_analysis(dfs, \
    #             n_resamples=n_resamples, destination_dir=destination_dir)



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(add_help=False)
    args, _ = arg_parser.parse_known_args()

    sys.exit(main(args))  # next section explains the use of sys.exit


