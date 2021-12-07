

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

from utils.custom_observer import CustomObserver
from utils.proj_setup import make_and_clear



import config.paths as paths
from corpus.tokenization import get_tokenizer, get_context

from layers.transformer_misc import get_length_percentiles
from spert_utils.config_setup import dict_to_config_file, get_prediction_file
from spert_utils.spert_io import merge_spert_files, spert2corpus

from spert_utils.config_setup import create_event_types_path, get_dataset_stats
from spert_utils.convert_brat import RELATION_DEFAULT
from scoring.scoring import score_docs

from spert_utils.spert_io import swap_type2subtype, map_type2subtype, plot_loss
from spert_utils.spert_scoring import score_spert_docs

import config.constants as constants




# Define experiment and load ingredients
ex = Experiment('step110_extraction')



@ex.config
def cfg():

    description = "unknown2"


    """
    Paths
    """
    source = "sdoh"
    source_file = os.path.join(paths.brat_import, source, constants.CORPUS_FILE)

    output_dir = paths.extraction

    mode = constants.TRAIN
    subdir = None

    if subdir is None:
        subdir = mode

    fast_run = True
    fast_count = 20 if fast_run else None

    destination = os.path.join(output_dir, subdir, description)
    if fast_run:
        destination += '_FAST_RUN'

    config_file = os.path.join(destination, "config.conf")

    train_subset = constants.TRAIN
    valid_subset = constants.DEV

    train_path = os.path.join(destination, 'data_train.json')
    valid_path = os.path.join(destination, 'data_valid.json')
    types_path = os.path.join(destination, "types.conf")

    # corpus preprocessing
    mapping = {}
    mapping["event_map"] = None
    mapping["relation_map"] = None
    mapping["tb_map"] = None
    mapping["attr_map"] = None

    # predict ANATOMY sub types with entity classifier
    types_config = {}
    types_config["relations"] = [RELATION_DEFAULT]
    types_config["entities"] = constants.ENTITIES
    types_config["subtypes"] = constants.SUBTYPES


    entity_types = constants.ENTITIES
    event_types = constants.EVENT_TYPES

    spert_path = '/home/lybarger/spert_plus/'

    """
    Model parameters
    """



    label = 'default'
    model_type = 'spert'
    model_path = "emilyalsentzer/Bio_ClinicalBERT" #'bert-base-cased'
    tokenizer_path = model_path

    train_batch_size = 15
    eval_batch_size = 2
    neg_entity_count = 100
    neg_relation_count = 100
    epochs = 1 if fast_run else 20
    lr = 5e-5
    lr_warmup = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0
    rel_filter_threshold = 0.4
    size_embedding = 25
    prop_drop = 0.2
    max_span_size = 10
    store_predictions = True
    store_examples = True
    sampling_processes = 4
    max_pairs = 1000
    final_eval = True
    log_path = f'{destination}/log/'
    save_path = f'{destination}/save/'

    subtype_classification = constants.CONCAT_LOGITS
    projection_size = 100
    projection_dropout = 0.0
    include_sent_task = False
    concat_sent_pred = False
    include_adjacent = False
    include_word_piece_task = False
    concat_word_piece_logits = False

    device = 3

    model_config = OrderedDict()
    model_config["label"] = label
    model_config["model_type"] = model_type
    model_config["model_path"] = model_path
    model_config["tokenizer_path"] = tokenizer_path
    model_config["train_path"] = train_path
    model_config["valid_path"] = valid_path
    model_config["types_path"] = types_path
    model_config["train_batch_size"] = train_batch_size
    model_config["eval_batch_size"] = eval_batch_size
    model_config["neg_entity_count"] = neg_entity_count
    model_config["neg_relation_count"] = neg_relation_count
    model_config["epochs"] = epochs
    model_config["lr"] = lr
    model_config["lr_warmup"] = lr_warmup
    model_config["weight_decay"] = weight_decay
    model_config["max_grad_norm"] = max_grad_norm
    model_config["rel_filter_threshold"] = rel_filter_threshold
    model_config["size_embedding"] = size_embedding
    model_config["prop_drop"] = prop_drop
    model_config["max_span_size"] = max_span_size
    model_config["store_predictions"] = store_predictions
    model_config["store_examples"] = store_examples
    model_config["sampling_processes"] = sampling_processes
    model_config["max_pairs"] = max_pairs
    model_config["final_eval"] = final_eval
    model_config["log_path"] = log_path
    model_config["save_path"] = save_path
    model_config["subtype_classification"] = subtype_classification
    model_config["projection_size"] = projection_size
    model_config["projection_dropout"] = projection_dropout
    model_config["include_sent_task"] = include_sent_task
    model_config["concat_sent_pred"] = concat_sent_pred
    model_config["include_adjacent"] = include_adjacent
    model_config["include_word_piece_task"] = include_word_piece_task
    model_config["concat_word_piece_logits"] = concat_word_piece_logits
    model_config["device"] = device

    # Scratch directory
    make_and_clear(destination)

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)

#
#
# def filter_for_anatomy(file, type=ANATOMY):
#     doc = json.load(open(file, 'r'))
#
#
#     for d in doc:
#         d["relations"] = []
#         assert len(d["entities"]) == len(d["subtypes"])
#         keep = [i for i, x in enumerate(d["entities"]) if x["type"] == type]
#         d["entities"] = [d["entities"][i] for i in keep]
#         d["subtypes"] = [d["subtypes"][i] for i in keep]
#
#     json.dump(doc, open(file, 'w'))


# def add_sent_labels(file, types, label_type=constants.TYPE):
#
#     doc = json.load(open(file, 'r'))
#
#     for d in doc:
#         sent_labels = {t:0 for t in types}
#         for entity in d["entities"]:
#             label = entity[label_type]
#             sent_labels[label] = 1
#         d['sent_labels'] = sent_labels
#
#     json.dump(doc, open(file, 'w'))

@ex.automain
def main(source_file, destination, config_file, model_config, spert_path, \
        types_config, mapping, fast_count,
        train_path, train_subset, valid_path, valid_subset, event_types,
        entity_types, types_path):

    '''
    Prepare spert inputs
    '''

    # load corpus
    corpus = joblib.load(source_file)

    # apply corpus mapping
    corpus.map_(**mapping, path=destination)

    # create formatted data
    for path, subset in [(train_path, train_subset), (valid_path, valid_subset)]:
        corpus.events2spert( \
                    include = subset,
                    event_types = event_types,
                    entity_types = entity_types,
                    path = path,
                    sample_count = fast_count)

        z = lsakdjflaskdjf


        swap_type2subtype(path, path)

        get_dataset_stats(dataset_path=path, dest_path=destination, name=subset)

    # create spert types file
    create_event_types_path(**types_config, path=types_path)

    # create configuration file
    dict_to_config_file(model_config, config_file)

    '''
    Call Spert
    '''
    logging.info("Destination = {}".format(destination))

    for dir in [model_config["log_path"], model_config["save_path"]]:
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)

    cwd = os.getcwd()
    os.chdir(spert_path)
    out = os.system(f'python ./spert.py train --config {config_file}')
    os.chdir(cwd)
    print("out", out)
    if out != 0:
        raise ValueError(f"python call error: {out}")
        assert False


    '''
    Post process output
    '''

    loss_csv_file = os.path.join(model_config["log_path"], 'loss_avg_train.csv')
    plot_loss(loss_csv_file, loss_column='loss_avg')

    loss_csv_file = os.path.join(model_config["log_path"], 'loss_train.csv')
    plot_loss(loss_csv_file, loss_column='loss')

    predict_file = os.path.join(model_config["log_path"], PREDICTIONS_JSON)


    map_type2subtype(predict_file, predict_file, map_=subtype2type)
    #if model_config["subtype_classification"] == NO_SUBTYPE:
    #    map_type2subtype(predict_file, predict_file, map_=subtype2type)
    #else:
    #    swap_type2subtype(predict_file, predict_file)


    merged_file = os.path.join(destination, PREDICTIONS_JSON)
    merge_spert_files(model_config["valid_path"], predict_file, merged_file)


    logging.info(f"Scoring predictions")
    logging.info(f"Gold file:                     {model_config['valid_path']}")
    logging.info(f"Prediction file, original:     {predict_file}")
    logging.info(f"Prediction file, merged_file:  {merged_file}")

    gold_docs = corpus.docs(include=valid_subset, as_dict=True)

    predict_corpus = spert2corpus(merged_file)
    predict_docs = predict_corpus.docs(as_dict=True)

    gold_docs = OrderedDict([(k, v) for k, v in gold_docs.items() if k in predict_docs])

    score_spert_docs(valid_path, merged_file, destination)

    score_docs(gold_docs, predict_docs, \
                            scoring = [EXACT, OVERLAP, PARTIAL, LABEL],
                            destination = destination)

    return 'Successful completion'
