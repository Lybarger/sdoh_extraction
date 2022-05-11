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
import config.constants as C
from config.constants import TRAIN, DEV, TEST, QC
from config.constants import LABELED_ARGUMENTS, REQUIRED_ARGUMENTS, EVENT_TYPES, CORPUS_FILE

# Define experiment and load ingredients
ex = Experiment('step013_tokenization_checks')

ID_FIELD = "id"
SUBSET_FIELD = "subset"
SOURCE_FIELD = "source"


@ex.config
def cfg():



    source = 'sdoh_challenge'

    source_dir = paths.brat_import
    source_file = os.path.join(source_dir, source, CORPUS_FILE)

    output_dir = os.path.join(paths.tokenization_checks, source)

    fast_run = False
    fast_count = 100 if fast_run else None

    subset = C.TRAIN


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
def main(destination, source_file, subset):

    tokenizer = get_tokenizer()

    # load corpus
    corpus = joblib.load(source_file)
    logging.info(f"Corpus loaded")



    docs = corpus.docs(as_dict=True, include=subset)

    output = []
    for id, doc in docs.items():
        output.append("")
        output.append("=" * 80)
        output.append(f"ID: {id}")
        output.append("-" * 80)
        output.append(doc.text)
        output.append("-" * 80)
        for line in doc.tokens:
            output.append(str(line))
        output.append("-" * 80)

    output = "\n".join(output)

    file = "tokenization_examples.txt"
    file = os.path.join(destination, file)
    with open(file, 'w') as f:
        f.write(output)



    return 'Successful completion'
