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





import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from corpus.corpus_brat import CorpusBrat
from utils.custom_observer import CustomObserver
from utils.proj_setup import make_and_clear

from corpus.tokenization import get_tokenizer

import config.paths as paths
from config.constants import TRAIN, DEV, TEST, QC
from config.constants import CORPUS_FILE

# Define experiment and load ingredients
ex = Experiment('step010_brat_import')

ID_FIELD = "id"
SUBSET_FIELD = "subset"
SOURCE_FIELD = "source"


# def tag_function_anno(id, annotator_position=0, subset_position=1, source_position=2):
#
#     parts = id.split(os.sep)
#
#     annotator = parts[annotator_position]
#
#     subset = parts[subset_position]
#     assert subset in [TRAIN, DEV, TEST, QC]
#
#     source = parts[source_position]
#
#     tags = set([annotator, subset, source])
#
#     return tags

def tag_function(id, subset_position=0, source_position=1):

    parts = id.split(os.sep)

    subset = parts[subset_position]
    assert subset in [TRAIN, DEV, TEST, QC]

    source = parts[source_position]

    tags = set([subset, source])

    return tags



@ex.config
def cfg():

    # source_name: str, name for saved corpus
    source_name = 'sdoh_challenge'

    # source_dir: str, directory with BRAT annotations
    source_dir = paths.sdoh_corpus_challenge

    # tag_func: function for assigning tags (e.g. train, dev. test) to documents
    tag_func = tag_function

    # fast_run: bool, if True, only import a subset of corpus for troubleshooting
    fast_run = False

    # fast_count: int, number of samples to import if fast_run==True. None imports all samples
    fast_count = 100 if fast_run else None

    # skip: list, list of files to exclude from import
    skip = None

    # corpus_object: obj, Corpus object for storing imported BRAT files
    corpus_object = CorpusBrat

    # output_dir: str, output directory for all BRAT imports
    output_dir = paths.brat_import

    # destination: str, output directory for imported corpus
    destination = os.path.join(output_dir,  source_name)
    if fast_run:
        destination += '_FAST_RUN'

    # Scratch directory
    make_and_clear(destination)

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)


@ex.automain
def main(destination, source_dir, corpus_object, fast_count, skip, tag_func):

    '''
    Events
    '''
    logging.info('Importing from:\t{}'.format(source_dir))
    corpus = corpus_object()
    corpus.import_dir(source_dir, \
                    n = fast_count,
                    skip = skip,
                    tag_function = tag_func)

    # Save annotated corpus
    logging.info('Saving corpus')
    fn_corpus = os.path.join(destination, CORPUS_FILE)
    joblib.dump(corpus, fn_corpus)

    return 'Successful completion'
