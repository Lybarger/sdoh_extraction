
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

EVENT_TYPES = [SC.ALCOHOL, SC.DRUG, SC.TOBACCO, SC.EMPLOYMENT, SC.LIVING_STATUS]

def get_p(row):

    nt = row[NT]
    np = row[NP]
    tp = row[TP]

    if np == 0:
        p = 0
    else:
        p = tp/np
    
    return p

def get_r(row):

    nt = row[NT]
    np = row[NP]
    tp = row[TP]

    if nt == 0:
        r = 0
    else:
        r = tp/nt
    
    return r

def get_f(row):

    nt = row[NT]
    np = row[NP]
    tp = row[TP]
    p = row[P]
    r = row[R]

    if p + r == 0:
        f1 = 0
    else:
        f1 = 2*p*r/(p + r)
    
    return f1



def analyze_file(source, destination_dir):
 
    df_all = pd.read_csv(source)

    df = df_all.groupby(['ID', EVENT])[[NT, NP, TP]].sum()
    df = df.reset_index()


    df[P] =  df.apply(get_p, axis=1)
    df[R] =  df.apply(get_r, axis=1)
    df[F1] = df.apply(get_f, axis=1)

    for event in EVENT_TYPES:

        df_event = df[df[EVENT] == event]
        df_event = df_event.sort_values([F1], ascending=True)


        f = os.path.join(destination_dir, f'{event}.csv')
        df_event.to_csv(f)
        
    




def main(args):

    destination_dir = paths.error_analysis_examples
    
    source = '/home/lybarger/sdoh_challenge/analyses/step219_error_analysis_score/scores.csv'


    create_project_folder(destination_dir)

    define_logging(destination_dir)


    logging.info("")
    logging.info('='*80)
    logging.info('step221_error_analysis_examples')
    logging.info('='*80)

    logging.info(f'source:              {source}')
    logging.info(f'destination_dir:     {destination_dir}')

        
    analyze_file( \
            source = source, 
            destination_dir = destination_dir)



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(add_help=False)
    args, _ = arg_parser.parse_known_args()

    sys.exit(main(args))  # next section explains the use of sys.exit


