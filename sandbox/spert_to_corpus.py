


import os
import re
import numpy as np
import json
import joblib
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#pd.set_option('display.max_rows', None)
from collections import Counter, OrderedDict
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil


from corpus.corpus_brat_spert import CorpusBratSpert
import config.constants as C



prediction_file = '/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/unknown2/data_valid.json'

transfer_argument_pairs = { \
        C.ALCOHOL:       C.STATUS_TIME,
        C.DRUG:          C.STATUS_TIME,
        C.TOBACCO:       C.STATUS_TIME,
        C.LIVING_STATUS: C.TYPE_LIVING,
        C.EMPLOYMENT:    C.STATUS_EMPLOY}

corpus = CorpusBratSpert()

corpus.import_spert_corpus(path=prediction_file, argument_pairs=transfer_argument_pairs)
