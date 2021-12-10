


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



prediction_file = '/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/unknown2/predictions.json'



corpus = CorpusBratSpert()

corpus.import_spert_corpus(path=prediction_file)
