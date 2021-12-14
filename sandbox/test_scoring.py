

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



from scoring.scoring import score_brat

import config.constants as C


gold_dir = "/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/unknown2_TRUNCATED/brat_true/"
predict_dir = "/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/unknown2_TRUNCATED/brat_true/"


df = score_brat(gold_dir, predict_dir, \
                            labeled_args = [C.STATUS_TIME, C.TYPE_LIVING, C.STATUS_EMPLOY],
                            score_trig = C.MIN_DIST,
                            score_span = C.PARTIAL,
                            score_labeled = C.OVERLAP,
                            path = '/home/lybarger/sdoh_challenge/repository/sandbox',
                            description = None)
print(df)
