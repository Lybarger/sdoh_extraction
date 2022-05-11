

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

from spert_utils.config_setup import create_event_types_path, get_dataset_stats
from spert_utils.convert_brat import RELATION_DEFAULT


from spert_utils.spert_io import plot_loss


import config.constants as C
from corpus.corpus_brat import CorpusBrat

# Define experiment and load ingredients
ex = Experiment('step111_extraction_multi_spert')


@ex.config
def cfg():

    # description: str describing run, which will be included in output path
    description = "default"

    # source_name: str defining source corpus name
    source_name = "sdoh_challenge"

    # source_file: str defining path to source_name corpus
    source_file = os.path.join(paths.brat_import, source_name, C.CORPUS_FILE)

    # mode: str defining mode from {"train"}
    mode = C.TRAIN

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
    output_dir = paths.extraction_multi

    # destination: str defining output path
    destination = os.path.join(output_dir, subdir, description)
    if fast_run:
        destination += '_FAST_RUN'

    # save_brat: bool indicating whether predictions should be saved an BRAT format
    #   save_brat == True for trouble shooting and final runs
        #   save_brat == False for most experimentation to avoid many small files
    save_brat = False

    # train_subset: str indicating the training subset
    #   likely train_subset == "train"
    train_subset = C.TRAIN

    # valid_subset: str indicating the validation subset
    #   likely valid_subset == "dev" or valid_subset == "test"
    valid_subset = C.DEV
    # valid_subset = train_subset

    '''
    Label definition
    '''
    label_definition = {}

    # subtype_default: str defining default (null) subtype value
    #   for consistency with SpERT should be "None"
    label_definition["subtype_default"] = C.SUBTYPE_DEFAULT

    # event_types: list of event types to include as entities
    #   ex. ["Lesion", "Medical_Problem"]
    label_definition["event_types"] = [C.ALCOHOL, C.DRUG, C.TOBACCO, C.LIVING_STATUS, C.EMPLOYMENT]

    # argument_types: list of argument types to include as entities
    #   ex. ["Anatomy", "Count", "Size"]
    label_definition["argument_types"] = [C.AMOUNT, C.DURATION, C.FREQUENCY, C.HISTORY, C.METHOD, C.TYPE]

    # entity_types: list of entities, including event types and argument types
    #   ex. ["Lesion", "Medical_Problem", "Anatomy", "Count", "Size"]
    label_definition["entity_types"] = label_definition["event_types"] + \
                                       label_definition["argument_types"]

    # subtype layers: list of subtype classification layers
    #   ex. ["Assertion", "Indication_Type", "Size"]
    label_definition["subtype_layers"] = [C.STATUS_TIME, C.STATUS_EMPLOY, C.TYPE_LIVING]

    # swapped_spans: list of arguments for which to the span should be
    # mapped to the trigger span
    #   ex. ["Assertion", "Indication_Type"]
    label_definition["swapped_spans"] = [C.STATUS_TIME, C.STATUS_EMPLOY, C.TYPE_LIVING]

    # skip_dup_trig: bool indicating whether duplicate trigger should be skipped
    label_definition["skip_dup_trig"] = True

    # args_by_event_type: dict defining the arguments associated with each event type
    #   ex. {"Indication": ["Assertion", "Indication_Type", ...],
    #         "Medical_problem": ["Assertion", "Anatomy", ...], ...}
    label_definition["args_by_event_type"] = C.ARGUMENTS_BY_EVENT_TYPE


    '''
    BRAT
    '''
    # attr_type_map: function for map text bound name to attribute name
    label_definition["attr_type_map"] = C.ATTR_TYPE_MAP

    # arg_role_map: dictionary for mapping argument roles
    label_definition["arg_role_map"] = C.ARGUMENT2ROLE

    '''
    Scoring
    '''
    # labeled_args: list of labeled arguments. Only used and scoring.
    #   NOTE: likely corresponds to the keys associated with spert_types_config["subtypes"]
    #   ex. ["Assertion", "Indication_Type", "Size"]
    label_definition["score_labeled_args"] = [C.STATUS_TIME, C.STATUS_EMPLOY, C.TYPE_LIVING]

    # argument types: list entity types, including "Trigger"
    # if None, all argument types included
    # ["Trigger", "Lesion", "Medical_Problem", "Anatomy", "Count", "Size"]
    label_definition["score_argument_types"] = None

    # role_map = None C.ARGUMENT2ROLE


    '''
    SpERT config
    '''
    # spert_types_config: dictionary defining the relation, entity, and
    # subtypes for the SpERT model
    spert_types_config = {}

    # spert_types_config["relations"]: list defining relation types
    #   NOTE: current implementation only supports a single relation type,
    #   words treats the relation classification as a binary task (connected vs not connected)
    spert_types_config["relations"] = [RELATION_DEFAULT]

    # spert_types_config["entities"]: list of entity types
    spert_types_config["entities"] = label_definition["entity_types"]

    # spert_types_config["subtypes"]: dict defining_subtype_layers
    #   dict keys define the subtype layers (arguments)
    #   dict values defines the list of label classes of each subtype layer (argument)
    #   ex. {"Assertion", ["present", "absent"], "Size", ["current", "past"]}
    spert_types_config["subtypes"] = OrderedDict()
    spert_types_config["subtypes"][C.STATUS_TIME] =     C.STATUS_TIME_CLASSES
    spert_types_config["subtypes"][C.STATUS_EMPLOY] =   C.STATUS_EMPLOY_CLASSES
    spert_types_config["subtypes"][C.TYPE_LIVING] =     C.TYPE_LIVING_CLASSES




    '''
    Scoring
    '''
    # Scoring:
    scoring = OrderedDict()
    scoring["trig_exact_span_exact"] =     dict(score_trig=C.EXACT,   score_span=C.EXACT,   score_labeled=C.LABEL)
    scoring["trig_overlap_span_exact"] =   dict(score_trig=C.OVERLAP, score_span=C.EXACT,   score_labeled=C.LABEL)
    scoring["trig_overlap_span_overlap"] = dict(score_trig=C.OVERLAP, score_span=C.OVERLAP, score_labeled=C.LABEL)




    """
    SpERT
    """

    # spert_path: str, path to spert code base
    username = os.getlogin()
    spert_path = f'/home/{username}/mspert/'



    # train_path: str, paths to data in spert format
    train_path = os.path.join(destination, 'data_train.json')
    valid_path = os.path.join(destination, 'data_valid.json')




    label = 'sdoh'
    model_type = 'spert'

    # model_path: str, name of pre-trained BERT model
    model_path = "emilyalsentzer/Bio_ClinicalBERT" #'bert-base-cased'

    # tokenizer_path: str, name of pre-trained BERT tokenizer
    tokenizer_path = model_path

    train_batch_size = 15
    eval_batch_size = 2
    neg_entity_count = 100
    neg_relation_count = 100
    epochs = 1 if fast_run else 1
    lr = 5e-5
    lr_warmup = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0
    rel_filter_threshold = 0.5
    size_embedding = 25
    prop_drop = 0.2
    max_span_size = 10
    store_predictions = True
    store_examples = True
    sampling_processes = 4
    max_pairs = 1000
    final_eval = True
    log_path = os.path.join(destination, 'log')
    save_path = os.path.join(destination,  'save')

    # config path: str, path for configuration file
    config_path = os.path.join(destination, "config.conf")

    # types path: str, path to spert label definition file
    types_path = os.path.join(destination, "types.conf")



    subtype_classification = C.CONCAT_LOGITS
    include_sent_task = False
    concat_sent_pred = False
    include_adjacent = False
    include_word_piece_task = False
    concat_word_piece_logits = False

    device = 0

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



@ex.automain
def main(source_file, destination, config_path, model_config, spert_path, \
        spert_types_config, fast_count,
        train_path, train_subset, valid_path, valid_subset,
        types_path,
        # argument_source, argument_target, default_subtype_value,

        scoring, save_brat, label_definition):




    '''
    Prepare spert inputs
    '''

    # load corpus
    corpus = joblib.load(source_file)


    # use trigger spans for all arguments and the list to use _trigger_span
    for arg in label_definition["swapped_spans"]:
        c = corpus.swap_spans( \
                        source = arg,
                        target = C.TRIGGER,
                        use_role = False)

    if valid_subset == train_subset:

        logging.warn('='*200 + f"\nValidation set and train set are equivalent\n" + '='*200)

    # create formatted data
    for path, subset, sample_count in [(train_path, train_subset, fast_count), (valid_path, valid_subset, fast_count)]:
        corpus.events2spert_multi( \
                    include = subset,
                    entity_types = label_definition["entity_types"],
                    subtype_layers = label_definition["subtype_layers"],
                    subtype_default = label_definition["subtype_default"],
                    path = path,
                    sample_count = sample_count,
                    include_doc_text = True)

        get_dataset_stats(dataset_path=path, dest_path=destination, name=subset)

    # create spert types file
    create_event_types_path(**spert_types_config, path=types_path)

    # create configuration file
    dict_to_config_file(model_config, config_path)

    '''
    Call Spert
    '''
    logging.info("Destination = {}".format(destination))

    for dir in [model_config["log_path"], model_config["save_path"]]:
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)


    cwd = os.getcwd()
    os.chdir(spert_path)
    out = os.system(f'python ./spert.py train --config {config_path}')
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

    predict_file = os.path.join(model_config["log_path"], C.PREDICTIONS_JSON)

    merged_file = os.path.join(destination, C.PREDICTIONS_JSON)
    merge_spert_files(model_config["valid_path"], predict_file, merged_file)


    logging.info(f"Scoring predictions")
    logging.info(f"Gold file:                     {model_config['valid_path']}")
    logging.info(f"Prediction file, original:     {predict_file}")
    logging.info(f"Prediction file, merged_file:  {merged_file}")


    gold_corpus = joblib.load(source_file)
    gold_docs = gold_corpus.docs(include=valid_subset, as_dict=True)

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
    predict_docs = predict_corpus.docs(as_dict=True)

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


    f = os.path.join(model_config["save_path"], C.LABEL_DEFINITION_FILE)
    joblib.dump(label_definition, f)

    return 'Successful completion'
