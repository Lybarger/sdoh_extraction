

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




import config.constants as C


gold_dir = "/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/sdoh_challenge_25_d2/brat_true/"
predict_dir = "/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/sdoh_challenge_25_d2/brat_predict/"
output_path = '/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/sdoh_challenge_25_d2/'

# df = score_brat(gold_dir, predict_dir, \
#                             labeled_args = [C.STATUS_TIME, C.TYPE_LIVING, C.STATUS_EMPLOY],
#                             score_trig = C.EXACT,
#                             score_span = C.EXACT,
#                             score_labeled = C.LABEL,
#                             path = path,
#                             description = 'trig_exact_span_exact')



from brat_scoring.scoring import score_brat_sdoh
from brat_scoring.constants import EXACT, LABEL, OVERLAP, PARTIAL, MIN_DIST

df = score_brat_sdoh( \
                gold_dir = gold_dir,
                predict_dir = predict_dir,
                output_path = output_path,
                score_trig = EXACT,
                score_span = EXACT,
                score_labeled = LABEL,
                include_detailed = True,
                loglevel = 'info',
                description = 'trig_exact_span_exact'
                )

df = score_brat_sdoh( \
                gold_dir = gold_dir,
                predict_dir = predict_dir,
                output_path = output_path,
                score_trig = OVERLAP,
                score_span = EXACT,
                score_labeled = LABEL,
                include_detailed = True,
                loglevel = 'info',
                description = 'trig_overlap_span_exact'
                )

df = score_brat_sdoh( \
                gold_dir = gold_dir,
                predict_dir = predict_dir,
                output_path = output_path,
                score_trig = OVERLAP,
                score_span = OVERLAP,
                score_labeled = LABEL,
                include_detailed = True,
                loglevel = 'info',
                description = 'trig_overlap_span_overlap'
                )

df = score_brat_sdoh( \
                gold_dir = gold_dir,
                predict_dir = predict_dir,
                output_path = output_path,
                score_trig = OVERLAP,
                score_span = PARTIAL,
                score_labeled = LABEL,
                include_detailed = True,
                loglevel = 'info',
                description = 'trig_overlap_span_partial'
                )
                
#
#
# df = score_brat(gold_dir, predict_dir, \
#                             labeled_args = [C.STATUS_TIME, C.TYPE_LIVING, C.STATUS_EMPLOY],
#                             score_trig = C.OVERLAP,
#                             score_span = C.EXACT,
#                             score_labeled = C.LABEL,
#                             path = path,
#                             description = 'trig_overlap_span_exact')
# print(df)
#
# df = score_brat(gold_dir, predict_dir, \
#                             labeled_args = [C.STATUS_TIME, C.TYPE_LIVING, C.STATUS_EMPLOY],
#                             score_trig = C.EXACT,
#                             score_span = C.PARTIAL,
#                             score_labeled = C.LABEL,
#                             path = path,
#                             description = 'trig_exact_span_partial')
# print(df)
#
#
# df = score_brat(gold_dir, predict_dir, \
#                             labeled_args = [C.STATUS_TIME, C.TYPE_LIVING, C.STATUS_EMPLOY],
#                             score_trig = C.OVERLAP,
#                             score_span = C.PARTIAL,
#                             score_labeled = C.LABEL,
#                             path = path,
#                             description = 'trig_overlap_span_partial')
# print(df)
#
#
# df = score_brat(gold_dir, predict_dir, \
#                             labeled_args = [C.STATUS_TIME, C.TYPE_LIVING, C.STATUS_EMPLOY],
#                             score_trig = C.OVERLAP,
#                             score_span = C.OVERLAP,
#                             score_labeled = C.LABEL,
#                             path = path,
#                             description = 'trig_overlap_span_overlap')
# print(df)
