from __future__ import division, print_function, unicode_literals

from sacred import Experiment
from sacred.observers import FileStorageObserver
import sys
import os
import re
import numpy as np
import json
import joblib
import pandas as pd
from collections import Counter, OrderedDict
import logging
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


#from corpus.corpus_brat import CorpusBrat
from corpus.corpus_brat_incidental import CorpusBratIncidental
from corpus.corpus_brat import CorpusBrat
from corpus.anatomy_norm import merge_anatomy_normalization

from utils.custom_observer import CustomObserver
from utils.proj_setup import make_and_clear
import config.constants as constants
import config.paths as paths
from corpus.tokenization import get_tokenizer

from utils.brat_import_incidental import get_label_freq, anatomy_histogram_orig, anatomy_histogram_merged, trig_span_histogram
from config.constants import TRAIN, DEV, TEST

# Define experiment and load ingredients
ex = Experiment('step010_brat_import')

ID_FIELD = "id"
SUBSET_FIELD = "subset"
SOURCE_FIELD = "source"

@ex.config
def cfg():



    source = 'radiology2'
    events_dir = paths.brat_radiology_events
    anatomy_dir = paths.brat_radiology_anatomy

    dir = paths.brat_import

    fast_run = False
    fast_count = 10 if fast_run else None

    skip = None

    anatomy_corpus_object = CorpusBrat
    corpus_object = CorpusBratIncidental
    rm_extra_lb = False
    snap_textbounds = False

    brat_dir = "radiology_anatomy"


    attr_map = {'Lesion-Maligancy-Value': "Lesion-Malignancy-Value"}


    splits_file = None if fast_run else '/home/lybarger/incidentalomas/resources/radiology_500_splits.csv'

    train_size = 0.7
    dev_size = 0.1
    test_size = 0.2

    shuffle = True
    random_state = 1

    subsets = [TRAIN, DEV, TEST]

    '''
    Paths
    '''
    if fast_run:
        destination = os.path.join(dir,  source+'_FAST_RUN')
    else:
        destination = os.path.join(dir,  source)


    # Scratch directory
    make_and_clear(destination)

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)








@ex.automain
def main(destination, events_dir, anatomy_dir, fast_run, corpus_object, anatomy_corpus_object, \
        fast_count, snap_textbounds, skip, attr_map, brat_dir, splits_file,
        train_size, dev_size, test_size, shuffle, random_state, subsets):

    tokenizer = get_tokenizer()

    '''
    Anatomy
    '''
    logging.info('Importing anatomy from:\t{}'.format(anatomy_dir))
    anatomy = anatomy_corpus_object()
    anatomy.import_dir(anatomy_dir, \
                    n = fast_count,
                    skip = skip)


    anatomy_histogram_orig(anatomy, path=destination)

    '''
    Events
    '''
    logging.info('Importing events from:\t{}'.format(events_dir))
    corpus = corpus_object()
    corpus.import_dir(events_dir, \
                    n = fast_count,
                    skip = skip)

    corpus.assign_splits( \
                train_size = train_size,
                dev_size = dev_size,
                test_size = test_size,
                splits_file = splits_file,
                random_state = random_state,
                shuffle = shuffle,
                path = destination,
                include = None,
                exclude = None)

    corpus.map_(attr_map=attr_map, path=destination)

    modifications = []
    for doc in corpus.docs():
        mods = merge_anatomy_normalization(doc, anatomy[doc.id])
        modifications.extend(mods)
    df = pd.DataFrame(modifications)
    f = os.path.join(destination, "anatomy_span_modification.csv")
    df.to_csv(f)

    anatomy_histogram_merged(corpus, path=destination, subset='train')

    trig_span_histogram(corpus, path=destination)

    if snap_textbounds:
        corpus.snap_textbounds()

    corpus.histogram(tokenizer=tokenizer, path=destination)
    corpus.quality_check(path=destination)
    corpus.annotation_summary(path=destination)
    corpus.label_summary(path=destination)
    corpus.tag_summary(path=destination)

    dir = os.path.join(destination, brat_dir)
    corpus.write_brat(path=dir)

    for subset in subsets:
        dir = os.path.join(destination, f'brat_{subset}')
        corpus.write_brat(path=dir, include=subset)


    # Save annotated corpus
    logging.info('Saving corpus')
    fn_corpus = os.path.join(destination, constants.CORPUS_FILE)
    joblib.dump(corpus, fn_corpus)


    return 'Successful completion'
