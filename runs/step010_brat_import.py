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
ex = Experiment('step010_brat_import')

ID_FIELD = "id"
SUBSET_FIELD = "subset"
SOURCE_FIELD = "source"


def tag_function_anno(id, annotator_position=0, subset_position=1, source_position=2):

    parts = id.split(os.sep)

    annotator = parts[annotator_position]

    subset = parts[subset_position]
    assert subset in [TRAIN, DEV, TEST, QC]

    source = parts[source_position]

    tags = set([annotator, subset, source])

    return tags

def tag_function(id, subset_position=0, source_position=1):

    parts = id.split(os.sep)

    subset = parts[subset_position]
    assert subset in [TRAIN, DEV, TEST, QC]

    source = parts[source_position]

    tags = set([subset, source])

    return tags



@ex.config
def cfg():



    source = 'sdoh_review'

    tag_func = tag_function
    if source == 'sdoh_review':
        source_dir = paths.sdoh_corpus_review
        tag_func = tag_function_anno

    elif source == 'sdoh_challenge':
        source_dir = paths.sdoh_corpus_challenge
    else:
        raise ValueError("invalid source")

    output_dir = paths.brat_import

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
def main(destination, source_dir, fast_run, corpus_object,  \
        fast_count, skip, brat_dir, annotator_position,
        labeled_arguments, required_arguments, id_pattern, event_types,
        tag_func):

    tokenizer = get_tokenizer()


    '''
    Events
    '''
    logging.info('Importing from:\t{}'.format(source_dir))
    corpus = corpus_object()
    corpus.import_dir(source_dir, \
                    n = fast_count,
                    skip = skip,
                    tag_function = tag_func)

    corpus.span_histogram(path=destination, entity_types=event_types)


    corpus.histogram(tokenizer=tokenizer, path=destination)

    corpus.quality_check( \
                    path = destination,
                    annotator_position = annotator_position,
                    labeled_arguments = labeled_arguments,
                    required_arguments = required_arguments,
                    id_pattern = id_pattern)

    corpus.annotation_summary(path=destination)
    corpus.label_summary(path=destination)
    corpus.tag_summary(path=destination)

    #dir = os.path.join(destination, brat_dir)
    #corpus.write_brat(path=dir)

    # Save annotated corpus
    logging.info('Saving corpus')
    fn_corpus = os.path.join(destination, CORPUS_FILE)
    joblib.dump(corpus, fn_corpus)


    return 'Successful completion'
