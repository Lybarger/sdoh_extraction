

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


from brat_scoring.scoring import score_docs


from utils.custom_observer import CustomObserver
from utils.proj_setup import make_and_clear

import config.paths as paths


from spert_utils.config_setup import dict_to_config_file, get_prediction_file
from spert_utils.spert_io import merge_spert_files

from spert_utils.config_setup import get_dataset_stats
from spert_utils.convert_brat import RELATION_DEFAULT


from spert_utils.spert_io import plot_loss


import config.constants as C
from corpus.corpus_brat import CorpusBrat

# Define experiment and load ingredients
ex = Experiment('step112_multi_spert_eval')


@ex.config
def cfg():

    # description: str describing run, which will be included in output path
    description = "default_text"

    # source_name: str defining source corpus name
    source_name = "sdoh_challenge"

    # mode: str defining mode from {"eval", "predict"}
    mode = C.PREDICT
    # mode = C.EVAL

    # source_file: str defining path to source corpus as pickled import of BRAT corpus
    #   ONLY used if mode == 'eval'
    # source_file = None
    source_file = os.path.join(paths.brat_import, source_name, C.CORPUS_FILE)

    # source_dir: str defining path to source directory with txt files
    #   ONLY used if mode == 'predict'
    # source_dir = '/home/lybarger/data/social_determinants_challenge_text/'
    source_dir = None

    # subdir: str defining subdirectory for run output
    subdir = None
    if subdir is None:
        subdir = mode

    # fast_run: bool indicating whether a full or fast run should be performed
    #   fast_run == True for basic trouble shooting
    #   fast_run == False for full (proper) experiments
    fast_run = False

    # fast count: int number of samples to use
    fast_count = 20 if fast_run else None

    # output_dir: str defining root output path
    output_dir = paths.multi_spert_eval

    # destination: str defining output path
    destination = os.path.join(output_dir, subdir, description)
    if fast_run:
        destination += '_FAST_RUN'

    # save_brat: bool indicating whether predictions should be saved an BRAT format
    #   save_brat == True for trouble shooting and final runs
        #   save_brat == False for most experimentation to avoid many small files
    save_brat = False

    # eval_subset: str indicating the validation subset
    #   likely eval_subset == "dev" or eval_subset == "test"
    eval_subset = C.DEV
    source_subset = C.UW
    subset = [eval_subset, source_subset]
    # eval_subset = None

    '''
    Scoring
    '''
    # Scoring:
    scoring = OrderedDict()
    scoring["trig_exact_span_exact"] =     dict(score_trig=C.EXACT,     score_span=C.EXACT,   score_labeled=C.LABEL)
    scoring["trig_overlap_span_exact"] =   dict(score_trig=C.OVERLAP,   score_span=C.EXACT,   score_labeled=C.LABEL)
    scoring["trig_min_dist_span_exact"] =  dict(score_trig=C.MIN_DIST,  score_span=C.EXACT,   score_labeled=C.LABEL)
    scoring["trig_overlap_span_overlap"] = dict(score_trig=C.OVERLAP,   score_span=C.OVERLAP, score_labeled=C.LABEL)

    """
    SpERT
    """

    # spert_path: str, path to spert code base
    username = os.getlogin()
    spert_path = f'/home/{username}/mspert/'

    # config path: str, path for configuration file
    config_path = os.path.join(destination, "config.conf")

    # train_path: str, paths to data in spert format
    dataset_path = os.path.join(destination, 'data_eval.json')


    # model_path: str, name of pre-trained BERT model
    model_path = "/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/train/sdoh_challenge_e10_d02/save" #'bert-base-cased'

    # tokenizer_path: str, name of pre-trained BERT tokenizer
    tokenizer_path = model_path

    eval_batch_size = 2
    log_path = f'{destination}/log/'
    device = 0
    rel_filter_threshold = 0.5
    max_span_size = 10
    store_predictions = True
    store_examples = True
    sampling_processes = 4
    max_pairs = 1000

    model_config = OrderedDict()
    model_config["model_path"] = model_path
    model_config["tokenizer_path"] = tokenizer_path
    model_config["dataset_path"] = dataset_path
    model_config["eval_batch_size"] = eval_batch_size
    model_config["rel_filter_threshold"] = rel_filter_threshold
    model_config["max_span_size"] = max_span_size
    model_config["store_predictions"] = store_predictions
    model_config["store_examples"] = store_examples
    model_config["sampling_processes"] = sampling_processes
    model_config["max_pairs"] = max_pairs
    model_config["log_path"] = log_path
    model_config["device"] = device


    # Scratch directory
    make_and_clear(destination)

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)



@ex.automain
def main(mode, source_file, source_dir, destination, config_path, model_config, spert_path, fast_count, \
        dataset_path, subset, scoring, save_brat):


    if mode == C.EVAL:
        assert source_file is not None,     f'''if mode == "eval", then source_file cannot be None'''
        assert os.path.exists(source_file), f'''source_file does not exist: {source_file}'''
        if source_dir is not None:
            logging.warn(f'''if mode == "eval", then source_dir must be None. ignoring source_dir''')

        # load corpus
        corpus = joblib.load(source_file)

    elif mode == C.PREDICT:
        if source_file is not None:
            logging.warn(f'''if mode == "predict", then source_file must be None. ignoring_source_file''')
        assert source_dir is not None,          f'''if mode == "predict", then source_dir cannot be None'''
        assert os.path.exists(source_dir),  f'''source_dir does not exist: {source_dir}'''

        corpus = CorpusBrat()
        corpus.import_text_dir(source_dir)

    else:
        raise ValueError(f"Invalid mode: {mode}")


    f = os.path.join(model_config["model_path"], C.LABEL_DEFINITION_FILE)
    label_definition = joblib.load(f)

    '''
    Prepare spert inputs
    '''

    # use trigger spans for all arguments and the list to use _trigger_span
    for arg in label_definition["swapped_spans"]:
        c = corpus.swap_spans( \
                        source = arg,
                        target = C.TRIGGER,
                        use_role = False)

    # create formatted data
    corpus.events2spert_multi( \
                include = subset,
                entity_types = label_definition["entity_types"],
                subtype_layers = label_definition["subtype_layers"],
                subtype_default = label_definition["subtype_default"],
                path = dataset_path,
                sample_count = fast_count,
                include_doc_text = True)

    get_dataset_stats(dataset_path=dataset_path, dest_path=destination, name=subset)


    # create configuration file
    dict_to_config_file(model_config, config_path)

    '''
    Call Spert
    '''
    logging.info("Destination = {}".format(destination))


    if os.path.exists(model_config["log_path"]) and os.path.isdir(model_config["log_path"]):
        shutil.rmtree(model_config["log_path"])

    cwd = os.getcwd()
    os.chdir(spert_path)
    out = os.system(f'python ./spert.py eval --config {config_path}')
    os.chdir(cwd)
    print("out", out)
    if out != 0:
        raise ValueError(f"python call error: {out}")
        assert False


    '''
    Post process output
    '''


    predict_file = os.path.join(model_config["log_path"], C.PREDICTIONS_JSON)

    merged_file = os.path.join(destination, C.PREDICTIONS_JSON)
    merge_spert_files(model_config["dataset_path"], predict_file, merged_file)


    logging.info(f"Scoring predictions")
    logging.info(f"Gold file:                     {model_config['dataset_path']}")
    logging.info(f"Prediction file, original:     {predict_file}")
    logging.info(f"Prediction file, merged_file:  {merged_file}")

    predict_corpus = CorpusBrat()
    predict_corpus.import_spert_corpus_multi( \
                                    path = merged_file,
                                    subtype_layers = label_definition["subtype_layers"],
                                    subtype_default = label_definition["subtype_default"],
                                    event_types = label_definition["event_types"],
                                    swapped_spans = label_definition["swapped_spans"],
                                    arg_role_map = label_definition["arg_role_map"],
                                    attr_type_map = label_definition["attr_type_map"],
                                    skip_dup_trig = label_definition["skip_dup_trig"])

    #predict_corpus.map_roles(role_map, path=destination)
    predict_corpus.prune_invalid_connections(label_definition["args_by_event_type"], path=destination)


    if mode == C.EVAL:

        predict_docs = predict_corpus.docs(as_dict=True)

        gold_corpus = joblib.load(source_file)
        gold_docs = gold_corpus.docs(include=subset, as_dict=True)

        for description, score_def in scoring.items():

            score_docs( \
                gold_docs = gold_docs,
                predict_docs = predict_docs,
                labeled_args = label_definition["score_labeled_args"], \
                score_trig = score_def["score_trig"],
                score_span = score_def["score_span"],
                score_labeled = score_def["score_labeled"],
                output_path = destination,
                description = description,
                argument_types = label_definition["score_argument_types"])

    elif mode == C.PREDICT:

        brat_dir = os.path.join(destination, "brat")

        predict_corpus.write_brat(brat_dir)


    else:
        raise ValueError(f"Invalid mode: {mode}")


    return 'Successful completion'
