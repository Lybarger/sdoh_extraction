
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

import numpy as np
from collections import OrderedDict, Counter
from tqdm import tqdm
import joblib

import os
import shutil
import logging

pd.set_option('display.max_rows', None)
import re


import config.paths as paths
import config.constants as C
from utils.path_utils import create_project_folder, define_logging

from brat_scoring.corpus import Corpus
# from brat_scoring.constants import SPACY_MODEL


def import_submission(ann_path, txt_path):

    logging.info(f'')
    logging.info(f'-'*80)
    logging.info(f'import_submission')
    logging.info(f'-'*80)    
    logging.info(f'\tann_path:        ({type(ann_path)}) - {ann_path}')
    logging.info(f'\ttxt_path:        ({type(txt_path)}) - {txt_path}')


    corpus = Corpus()
    summary, detailed = corpus.import_predictions( \
                    txt_path = txt_path,
                    ann_path = ann_path,
                    strict_import = False)

    docs = corpus.docs(as_dict=True)

    logging.info(f'\tdoc count:       {len(docs)}')

    event_counts = Counter()
    for k, doc in docs.items():
        subset, source, num = k.split('/')
        event_counts[(subset, source)] += len(doc.events())

    logging.info(f'\tevent count:')
    for k, c in event_counts.items():
        logging.info(f'\t\t{k} - {c}')

    return (corpus, summary, detailed)

def import_submissions(destination_dir, predict_dir, gold_dir):


    logging.info(f'-'*80)
    logging.info(f'score_submissions')
    logging.info(f'-'*80)
    logging.info('')
    logging.info(f'INPUTS:')
    logging.info(f'destination_dir:         {destination_dir}')
    logging.info(f'predict_dir:             {predict_dir}')
    logging.info(f'gold_dir:                {gold_dir}')
    logging.info('')



    logging.info("")
    logging.info(f"Gold import")
    gold_corpus, summary_, detailed_ = import_submission(ann_path=gold_dir, txt_path=gold_dir)
 
    f = os.path.join(destination_dir, 'gold.pkl')
    joblib.dump(gold_corpus, f)
    logging.info(f"Gold saved:              {f}")


    logging.info(f"Predict import")
    predict_corpus, summary_, detailed_ = import_submission(ann_path=predict_dir, txt_path=gold_dir)

    f = os.path.join(destination_dir, 'predict.pkl')
    joblib.dump(predict_corpus, f)
    logging.info(f"Predict saved:           {f}")

    return True

def main(args):

    destination_dir = paths.error_analysis_import

    create_project_folder(destination_dir)

    define_logging(destination_dir)

    gold_dir = '/home/lybarger/n2c2_2022_sdoh/data/c/'
    predict_dir = '/home/lybarger/sdoh_challenge/analyses/step112_multi_spert_infer/subtask_c/sdoh_challenge_e14_d02/brat/'


    logging.info("")
    logging.info('='*80)
    logging.info('step218_error_analysis_import')
    logging.info('='*80)

    logging.info(f'destination_dir:     {destination_dir}')
    logging.info(f'gold_dir:            {gold_dir}')
    logging.info(f'predict_dir:         {predict_dir}')



    import_submissions( \
                destination_dir = destination_dir,
                predict_dir = predict_dir,
                gold_dir = gold_dir)



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--fast_run', action='store_true')
    args, _ = arg_parser.parse_known_args()

    sys.exit(main(args))  # next section explains the use of sys.exit
