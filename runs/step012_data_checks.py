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
from config.constants import LABELED_ARGUMENTS, REQUIRED_ARGUMENTS, EVENT_TYPES, CORPUS_FILE

# Define experiment and load ingredients
ex = Experiment('step012_data_checks')

ID_FIELD = "id"
SUBSET_FIELD = "subset"
SOURCE_FIELD = "source"


@ex.config
def cfg():



    source = 'sdoh_challenge'

    source_dir = paths.brat_import
    source_file = os.path.join(source_dir, source, CORPUS_FILE)

    output_dir = os.path.join(paths.data_checks, source)

    fast_run = False
    fast_count = 100 if fast_run else None
    annotator_position = 0
    labeled_arguments = LABELED_ARGUMENTS
    required_arguments = REQUIRED_ARGUMENTS
    event_types = EVENT_TYPES

    id_pattern = None

    skip = None

    corpus_object = CorpusBrat

    brat_dir = "brat"

    '''
    Paths
    '''
    if fast_run:
        destination = os.path.join(output_dir,  source+'_FAST_RUN')
    else:
        destination = os.path.join(output_dir,  source)

    # Scratch directory
    make_and_clear(destination)

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)




@ex.automain
def main(destination, source_file, fast_run,  \
        labeled_arguments, required_arguments, id_pattern, event_types):

    tokenizer = get_tokenizer()

    # load corpus
    corpus = joblib.load(source_file)
    logging.info(f"Corpus loaded")


    # corpus.span_histogram(path=destination, entity_types=event_types)
    # logging.info(f"Span histograms created")
    #
    # corpus.histogram(tokenizer=tokenizer, path=destination)
    # logging.info(f"Histogram created")

    corpus.quality_check( \
                    path = destination,
                    labeled_arguments = labeled_arguments,
                    required_arguments = required_arguments,
                    id_pattern = id_pattern)
    logging.info(f"Quality checks performed")


    corpus.duplicate_check(path = destination)



    corpus.annotation_summary(path=destination)
    logging.info(f"Annotation summary created")

    corpus.label_summary(path=destination)
    logging.info(f"Label summary created")

    corpus.tag_summary(path=destination)
    logging.info(f"Tag summary created")

    return 'Successful completion'
