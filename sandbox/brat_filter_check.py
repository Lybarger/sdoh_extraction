


from __future__ import division, print_function, unicode_literals

from sacred import Experiment
from sacred.observers import FileStorageObserver

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


from corpus.corpus_brat import CorpusBrat
from utils.proj_setup import make_and_clear


source = "/home/lybarger/sdoh_challenge/analyses/step010_brat_import/sdoh/corpus.pkl"

destination = "/home/lybarger/sdoh_challenge/analyses/sandbox/brat_filt_check/"

event_types = C.EVENT_TYPES
argument_types = C.EVENT_TYPES + C.LABELED_ARGUMENTS + C.SPAN_ONLY_ARGUMENTS




original = os.path.join(destination, "original")
filtered = os.path.join(destination, "filtered")
make_and_clear(original)
make_and_clear(filtered)

corpus = joblib.load(source)



corpus.label_summary(path=original)


brat_true = os.path.join(filtered, "brat")
corpus.write_brat(path=brat_true, \
                event_types=event_types,
                argument_types=argument_types)

filtered_corpus = CorpusBrat()
filtered_corpus.import_dir(brat_true)
filtered_corpus.label_summary(path=filtered)
