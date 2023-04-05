from __future__ import division, print_function, unicode_literals

import sys
import os
import joblib
import logging
import argparse

from corpus.corpus_brat import CorpusBrat
from config.constants import TRAIN, DEV, TEST, QC, TRAIN_DEV

'''
Example usage

python import_corpus.py --source /home/lybarger/data/social_determinants_challenge/ --output_file /home/lybarger/sdoh_challenge/output/corpus.pkl

'''


def tag_function(id, subset_position=0, source_position=1):

    parts = id.split(os.sep)

    subset = parts[subset_position]
    assert subset in [TRAIN, DEV, TEST, QC]

    source = parts[source_position]

    tags = set([subset, source])

    if (TRAIN in tags) or (DEV in tags):
        tags.add(TRAIN_DEV)


    return tags


def main(args):

    '''
    Events
    '''
    logging.info(f'Importing from:\t{args.source}')
    corpus = CorpusBrat()
    corpus.import_dir(path = args.source, \
                    tag_function = tag_function)

    # Save annotated corpus
    logging.info('Saving corpus')
    joblib.dump(corpus, args.output_file)

    return True


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--source', type=str, help="input directory with SHAC annotations")
    arg_parser.add_argument('--output_file', type=str, help="output file")

    args, _ = arg_parser.parse_known_args()

    sys.exit(main(args)) 
