from __future__ import division, print_function, unicode_literals

import os
import re
import numpy as np
import json
import joblib
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from collections import OrderedDict
import logging
from sklearn.model_selection import train_test_split
import sys
import shutil
import argparse


from brat_scoring.scoring import score_docs, micro_average_subtypes

import config.paths as paths

from utils.misc import get_include
from spert_utils.config_setup import dict_to_config_file, get_prediction_file
from spert_utils.spert_io import merge_spert_files
from spert_utils.config_setup import get_dataset_stats
import config.constants as C
from corpus.corpus_brat import CorpusBrat


'''

python infer_mspert.py --source_file /home/lybarger/sdoh_challenge/output/corpus.pkl --destination /home/lybarger/sdoh_challenge/output/eval/ --mspert_path /home/lybarger/mspert/ --mode eval --eval_subset test --eval_source uw --model_path /home/lybarger/sdoh_challenge/output/model10/save/ --device 0


python infer_mspert.py --source_file /home/lybarger/sdoh_challenge/output2/corpus.pkl --destination /home/lybarger/sdoh_challenge/output2/eval/ --mspert_path /home/lybarger/mspert/ --mode eval --eval_subset test --eval_source uw --model_path /home/lybarger/sdoh_challenge/output/model10/save/ --device 0


python infer_mspert.py --source_dir /home/lybarger/data/social_determinants_challenge_text/ --destination /home/lybarger/sdoh_challenge/output/predict/ --mspert_path /home/lybarger/mspert/ --mode predict --model_path /home/lybarger/sdoh_challenge/output/model10/save/ --device 0

'''


def get_scoring_def():
    '''
    Scoring
    '''
    # Scoring:
    scoring = OrderedDict()
    scoring["n2c2"] = dict(score_trig=C.OVERLAP, score_span=C.EXACT, score_labeled=C.LABEL)
    return scoring   


def main(args):


    # config path: str, path for configuration file
    config_path = os.path.join(args.destination, "config.conf")

    # train_path: str, paths to data in spert format
    dataset_path = os.path.join(args.destination, 'data_eval.json')

    log_path = f'{args.destination}/log/'

    eval_include = get_include([args.eval_subset, args.eval_source])

    scoring = get_scoring_def()

    model_config = OrderedDict()
    model_config["model_path"] = args.model_path
    model_config["tokenizer_path"] = args.model_path
    model_config["eval_batch_size"] = args.eval_batch_size
    model_config["rel_filter_threshold"] = args.rel_filter_threshold
    model_config["max_span_size"] = args.max_span_size
    model_config["store_predictions"] = args.store_predictions
    model_config["store_examples"] = args.store_examples
    model_config["sampling_processes"] = args.sampling_processes
    model_config["max_pairs"] = args.max_pairs
    model_config["no_overlapping"] = args.no_overlapping
    model_config["device"] = args.device

    model_config["dataset_path"] = dataset_path
    model_config["log_path"] = log_path


    if args.mode == C.EVAL:
        assert args.source_file is not None, f'''if mode == "eval", then args.source_file cannot be None'''
        assert os.path.exists(args.source_file), f'''args.source_file does not exist: {args.source_file}'''
        if args.source_dir is not None:
            logging.warn(f'''if mode == "eval", then args.source_dir must be None. ignoring args.source_dir''')

        # load corpus
        corpus = joblib.load(args.source_file)

    elif args.mode == C.PREDICT:
        if args.source_file is not None:
            logging.warn(f'''if mode == "predict", then args.source_file must be None. ignoring_args.source_file''')
        assert args.source_dir is not None,          f'''if mode == "predict", then args.source_dir cannot be None'''
        assert os.path.exists(args.source_dir),  f'''args.source_dir does not exist: {args.source_dir}'''

        corpus = CorpusBrat()
        corpus.import_text_dir(args.source_dir)

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
    fast_count = args.fast_count if args.fast_run else None
    corpus.events2spert_multi( \
                include = eval_include,
                entity_types = label_definition["entity_types"],
                subtype_layers = label_definition["subtype_layers"],
                subtype_default = label_definition["subtype_default"],
                path = dataset_path,
                sample_count = fast_count,
                include_doc_text = True)

    if eval_include is None:
        include_name = 'None'
    else:
        include_name = '_'.join(list([x for x in eval_include if x is not None]))
    get_dataset_stats(dataset_path=dataset_path, dest_path=args.destination, name=include_name)

    # create configuration file
    dict_to_config_file(model_config, config_path)

    '''
    Call Spert
    '''
    logging.info("Destination = {}".format(args.destination))

    if os.path.exists(model_config["log_path"]) and os.path.isdir(model_config["log_path"]):
        shutil.rmtree(model_config["log_path"])

    cwd = os.getcwd()
    os.chdir(args.mspert_path)
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

    merged_file = os.path.join(args.destination, C.PREDICTIONS_JSON)
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
    predict_corpus.prune_invalid_connections(label_definition["args_by_event_type"], path=args.destination)


    if args.mode == C.EVAL:

        predict_docs = predict_corpus.docs(as_dict=True)

        gold_corpus = joblib.load(args.source_file)
        gold_docs = gold_corpus.docs(include=eval_include, as_dict=True)

        for description, score_def in scoring.items():

            df = score_docs( \
                gold_docs = gold_docs,
                predict_docs = predict_docs,
                labeled_args = label_definition["score_labeled_args"], \
                score_trig = score_def["score_trig"],
                score_span = score_def["score_span"],
                score_labeled = score_def["score_labeled"],
                output_path = args.destination,
                description = description,
                argument_types = label_definition["score_argument_types"])

            df = micro_average_subtypes(df)
            f = os.path.join(args.destination, f"scores_{description}_micro.csv")
            df = df.to_csv(f, index=False)

    elif args.mode == C.PREDICT:

        pass

    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    if args.save_brat:
        brat_dir = os.path.join(args.destination, "brat")
        logging.info("fSaving brat: {brat_dir}")
        predict_corpus.write_brat(brat_dir)

    return 'Successful completion'




if __name__ == '__main__':


    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--source_file', type=str, help="path to input corpus object")
    arg_parser.add_argument('--source_dir', type=str, help="path to input directory of unlabeled text")
    arg_parser.add_argument('--destination', type=str, help="path to output directory", required=True)
    arg_parser.add_argument('--mspert_path', type=str, help="path to mspert", required=True)
    arg_parser.add_argument('--mode', type=str, default='eval', help="inference mode: 'eval' for assessing performance against labeled data and 'predict' for applying extractor to label text without evaluation", required=True)
    arg_parser.add_argument('--fast_run', default=False, action='store_true', help="only train a small portion of training set for debugging")
    arg_parser.add_argument('--fast_count', type=int, default=20, help="")
    arg_parser.add_argument('--eval_subset', type=str, default=None, help="tag for evaluation subset from {train, dev, test, None}")
    arg_parser.add_argument('--eval_source', type=str, default=None, help="tag for evaluation source from {None, 'uw', 'mimic'}. None will use both uw and mimic")
    arg_parser.add_argument('--model_path',     type=str, help="fine-tuned mspert model", required=True)
    arg_parser.add_argument('--eval_batch_size', type=int, default=2, help="evaluation batch size")
    arg_parser.add_argument('--rel_filter_threshold', type=float, default=0.5, help="relation filter threshold")
    arg_parser.add_argument('--size_embedding', type=int, default=25, help="size for size embeddings")
    arg_parser.add_argument('--prop_drop', type=float, default=0.2, help="dropout")
    arg_parser.add_argument('--max_span_size', type=int, default=10, help="maximum span size")
    arg_parser.add_argument('--store_predictions', default=True,  action='store_false', help="store predictions?")
    arg_parser.add_argument('--store_examples', default=True,  action='store_false', help="store examples?")
    arg_parser.add_argument('--sampling_processes', type=int, default=4, help="number of sampling processes")
    arg_parser.add_argument('--max_pairs', type=int, default=1000, help="maximum relation pairs")
    arg_parser.add_argument('--no_overlapping', default=True, action='store_false', help="disallow overlapping spans")                
    arg_parser.add_argument('--device', type=int, default=0, help="GPU device")    
    arg_parser.add_argument('--save_brat', default=True, action='store_false', help="save predictions in brat format")                
    args, _ = arg_parser.parse_known_args()

    sys.exit(main(args)) 
