




from __future__ import division, print_function, unicode_literals
import sys
import argparse
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
import shutil


from brat_scoring.scoring import score_docs

from utils.misc import get_include
import config.paths as paths

from spert_utils.config_setup import dict_to_config_file, get_prediction_file
from spert_utils.spert_io import merge_spert_files

from spert_utils.config_setup import create_event_types_path, get_dataset_stats
from spert_utils.convert_brat import RELATION_DEFAULT


from spert_utils.spert_io import plot_loss


import config.constants as C
from corpus.corpus_brat import CorpusBrat



'''
python train_mspert.py --source_file /home/lybarger/sdoh_challenge/output/corpus.pkl  --destination /home/lybarger/sdoh_challenge/output/model/ --mspert_path /home/lybarger/mspert/     --model_path "emilyalsentzer/Bio_ClinicalBERT" --tokenizer_path "emilyalsentzer/Bio_ClinicalBERT" --epochs 10 --train_subset train --valid_subset dev --train_source None --valid_source uw


python train_mspert.py --source_file /home/lybarger/sdoh_challenge/output/corpus.pkl  --destination /home/lybarger/sdoh_challenge/output2/model01/ --mspert_path /home/lybarger/mspert/     --model_path "emilyalsentzer/Bio_ClinicalBERT" --tokenizer_path "emilyalsentzer/Bio_ClinicalBERT" --epochs 1 --train_subset train --valid_subset dev --train_source None --valid_source uw

python train_mspert.py --source_file /home/lybarger/sdoh_challenge/output2/corpus.pkl  --destination /home/lybarger/sdoh_challenge/output2/model01/ --mspert_path /home/lybarger/mspert/     --model_path "emilyalsentzer/Bio_ClinicalBERT" --tokenizer_path "emilyalsentzer/Bio_ClinicalBERT" --epochs 1 --train_subset train --valid_subset dev --train_source None --valid_source uw

'''

def get_label_definition():
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

    return label_definition


def get_scoring_def():
    '''
    Scoring
    '''
    # Scoring:
    scoring = OrderedDict()
    scoring["n2c2"] = dict(score_trig=C.OVERLAP, score_span=C.EXACT, score_labeled=C.LABEL)
    return scoring    


def get_spert_config(label_definition):

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


    return spert_types_config

def main(args):

    log_path = os.path.join(args.destination, 'log')
    save_path = os.path.join(args.destination,  'save')

    # train_path: str, paths to data in spert format
    train_path = os.path.join(args.destination, 'data_train.json')
    valid_path = os.path.join(args.destination, 'data_valid.json')

    # config path: str, path for configuration file
    config_path = os.path.join(args.destination, "config.conf")

    # types path: str, path to spert label definition file
    types_path = os.path.join(args.destination, "types.conf")


    model_config = OrderedDict()
    model_config["model_path"] = args.model_path
    model_config["tokenizer_path"] = args.tokenizer_path
    model_config["train_batch_size"] = args.train_batch_size
    model_config["eval_batch_size"] = args.eval_batch_size
    model_config["neg_entity_count"] = args.neg_entity_count
    model_config["neg_relation_count"] = args.neg_relation_count
    model_config["epochs"] = args.epochs
    model_config["lr"] = args.lr
    model_config["lr_warmup"] = args.lr_warmup
    model_config["weight_decay"] = args.weight_decay
    model_config["max_grad_norm"] = args.max_grad_norm
    model_config["rel_filter_threshold"] = args.rel_filter_threshold
    model_config["size_embedding"] = args.size_embedding
    model_config["prop_drop"] = args.prop_drop
    model_config["max_span_size"] = args.max_span_size
    model_config["store_predictions"] = args.store_predictions
    model_config["store_examples"] = args.store_examples
    model_config["sampling_processes"] = args.sampling_processes
    model_config["max_pairs"] = args.max_pairs
    model_config["final_eval"] = args.final_eval
    model_config["no_overlapping"] = args.no_overlapping
    model_config["device"] = args.device

    model_config["train_path"] = train_path
    model_config["valid_path"] = valid_path
    model_config["types_path"] = types_path
    model_config["log_path"] = log_path
    model_config["save_path"] = save_path

    model_config["subtype_classification"] = C.CONCAT_LOGITS
    model_config["label"] = 'sdoh'
    model_config["model_type"] = 'spert'
    model_config["include_sent_task"] = False
    model_config["concat_sent_pred"] = False
    model_config["include_adjacent"] = False
    model_config["include_word_piece_task"] = False
    model_config["concat_word_piece_logits"] = False


    label_definition = get_label_definition()
    spert_types_config = get_spert_config(label_definition)
    scoring = get_scoring_def()


    # combine source and subset
    train_include = get_include([args.train_subset, args.train_source])
    valid_include = get_include([args.valid_subset, args.valid_source])


    '''
    Prepare spert inputs
    '''

    # load corpus
    corpus = joblib.load(args.source_file)


    # use trigger spans for all arguments and the list to use _trigger_span
    for arg in label_definition["swapped_spans"]:
        c = corpus.swap_spans( \
                        source = arg,
                        target = C.TRIGGER,
                        use_role = False)

    if valid_include == train_include:
        logging.warn('='*200 + f"\nValidation set and train set are equivalent\n" + '='*200)

    # create formatted data

    fast_count = args.fast_count if args.fast_run else None
    for path, include, sample_count in [(train_path, train_include, fast_count), (valid_path, valid_include, fast_count)]:
        corpus.events2spert_multi( \
                    include = include,
                    entity_types = label_definition["entity_types"],
                    subtype_layers = label_definition["subtype_layers"],
                    subtype_default = label_definition["subtype_default"],
                    path = path,
                    sample_count = sample_count,
                    include_doc_text = True)

        include_name = '_'.join(list(include))
        get_dataset_stats(dataset_path=path, dest_path=args.destination, name=include_name)

    # create spert types file
    create_event_types_path(**spert_types_config, path=types_path)

    # create configuration file
    dict_to_config_file(model_config, config_path)

    '''
    Call Spert
    '''
    logging.info("Destination = {}".format(args.destination))

    for dir in [model_config["log_path"], model_config["save_path"]]:
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)


    cwd = os.getcwd()
    os.chdir(args.mspert_path)
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

    merged_file = os.path.join(args.destination, C.PREDICTIONS_JSON)
    merge_spert_files(model_config["valid_path"], predict_file, merged_file)


    logging.info(f"Scoring predictions")
    logging.info(f"Gold file:                     {model_config['valid_path']}")
    logging.info(f"Prediction file, original:     {predict_file}")
    logging.info(f"Prediction file, merged_file:  {merged_file}")


    gold_corpus = joblib.load(args.source_file)
    gold_docs = gold_corpus.docs(include=valid_include, as_dict=True)

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

    #predict_corpus.map_roles(role_map, path=args.destination)
    predict_corpus.prune_invalid_connections(label_definition["args_by_event_type"], path=args.destination)
    predict_docs = predict_corpus.docs(as_dict=True)

    for description, score_def in scoring.items():

        score_docs( \
            gold_docs = gold_docs,
            predict_docs = predict_docs,
            labeled_args = label_definition["score_labeled_args"], \
            score_trig = score_def["score_trig"],
            score_span = score_def["score_span"],
            score_labeled = score_def["score_labeled"],
            output_path = args.destination,
            description = description,
            argument_types = label_definition["score_argument_types"])


    f = os.path.join(model_config["save_path"], C.LABEL_DEFINITION_FILE)
    joblib.dump(label_definition, f)

    return 'Successful completion'


if __name__ == '__main__':

    # 

    """
    SpERT
    """
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--source_file', type=str, help="path to input corpus object", required=True)
    arg_parser.add_argument('--destination', type=str, help="path to output directory", required=True)
    arg_parser.add_argument('--mspert_path', type=str, help="path to mspert", required=True)

    arg_parser.add_argument('--fast_run', default=False, action='store_true', help="only train a small portion of training set for debugging")
    arg_parser.add_argument('--fast_count', type=int, default=20, help="")
    arg_parser.add_argument('--train_subset', type=str, default='train', help="tag for training subset from {train, dev, test}")
    arg_parser.add_argument('--valid_subset', type=str, default='dev', help="tag for validation subset from {train, dev, test}")
    arg_parser.add_argument('--train_source', type=str, help="tag for training soruce from {None, 'uw', 'mimic'}. None will use both uw and mimic")
    arg_parser.add_argument('--valid_source', type=str, help="tag for validation soruce from {None, 'uw', 'mimic'}. None will use both uw and mimic")

    arg_parser.add_argument('--model_path',     type=str, default="emilyalsentzer/Bio_ClinicalBERT", help="pretrained BERT model")
    arg_parser.add_argument('--tokenizer_path', type=str, default="emilyalsentzer/Bio_ClinicalBERT",help="pretrained BERT tokenizer")  
    arg_parser.add_argument('--train_batch_size', type=int, default=15, help="training batch size")
    arg_parser.add_argument('--eval_batch_size', type=int, default=2, help="evaluation batch size")
    arg_parser.add_argument('--neg_entity_count', type=int, default=100, help="negative entity County")
    arg_parser.add_argument('--neg_relation_count', type=int, default=100, help="negative relation count")
    arg_parser.add_argument('--epochs', type=int, default=1, help="number of epochs")
    arg_parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    arg_parser.add_argument('--lr_warmup', type=float, default=0.1, help="learning rate warm-up")
    arg_parser.add_argument('--weight_decay', type=float, default=0.01, help="learning we decay")
    arg_parser.add_argument('--max_grad_norm', type=float, default=1.0, help="maximum gradient norm")
    arg_parser.add_argument('--rel_filter_threshold', type=float, default=0.5, help="relation filter threshold")
    arg_parser.add_argument('--size_embedding', type=int, default=25, help="size for size embeddings")
    arg_parser.add_argument('--prop_drop', type=float, default=0.2, help="dropout")
    arg_parser.add_argument('--max_span_size', type=int, default=10, help="maximum span size")
    arg_parser.add_argument('--store_predictions', default=True, action='store_false', help="store predictions?")
    arg_parser.add_argument('--store_examples', default=True, action='store_false', help="store examples?")
    arg_parser.add_argument('--sampling_processes', type=int, default=4, help="number of sampling processes")
    arg_parser.add_argument('--max_pairs', type=int, default=1000, help="maximum relation pairs")
    arg_parser.add_argument('--final_eval', default=True, action='store_false', help="perform final evaluation?")
    arg_parser.add_argument('--no_overlapping', default=True, action='store_false', help="disallow overlapping spans")                
    arg_parser.add_argument('--device', type=int, default=0, help="GPU device")    
    
    args, _ = arg_parser.parse_known_args()

    sys.exit(main(args)) 