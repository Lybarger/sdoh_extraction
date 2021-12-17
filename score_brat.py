

import argparse
import os
import sys

import config.constants as C

from scoring.scoring import score_brat

def get_argparser():
    parser = argparse.ArgumentParser(description = 'compares and scores two directories of brat files and summaries the performance in a csv file')
    parser.add_argument('gold_dir',        type=str, help="path to input directory with gold labels in BRAT format")
    parser.add_argument('predict_dir',     type=str, help="path to input directory with predicted labels in BRAT format")
    parser.add_argument('output',          type=str, help="path to output csv file")
    parser.add_argument('--labeled_args', type=str, default=[C.STATUS_TIME, C.TYPE_LIVING, C.STATUS_EMPLOY], nargs='+', help=f'span only arguments')
    parser.add_argument('--score_trig',    type=str, default=C.EXACT, help=f'equivalence criteria for triggers, {{{C.EXACT}, {C.OVERLAP}, {C.MIN_DIST}}}')
    parser.add_argument('--score_span',    type=str, default=C.EXACT, help=f'equivalence criteria for span only arguments, {{{C.EXACT}, {C.OVERLAP}, {C.PARTIAL}}}')
    parser.add_argument('--score_labeled', type=str, default=C.LABEL, help=f'equivalence criteria for labeled arguments (span with value arguments), {{{C.EXACT}, {C.OVERLAP}, {C.LABEL}}}')
    return parser

def main(args):
    '''
    This function scores a set of labels in BRAT format, relative to a set of
    gold labels also in BRAT format. The scores are saved in a comma separated
    values (CSV) file with the following columns:

    event - events type, like Alcohol, Drug, Employment, etc.
    argument - event trigger and argument, like Trigger, StatusTime, History, etc.
    subtype	- subtype labels for labeled arguments, like current or past for StatusTime
    NT - count of true (gold) labels
    NP - count of predicted labels
    TP - counted true positives
    P - precision
    R - recall
    F1 - f-1 score - harmonic mean of precision and recall

    Example (without commas for readability):
    event         argument             subtype             NT      NP      TP      P       R       F1
    Alcohol       StatusTime           current             63      47      23     0.49    0.37    0.42
    Alcohol       StatusTime           none                68      98      53     0.54    0.78    0.64
    Alcohol       StatusTime           past                31      3       1      0.33    0.03    0.06
    Alcohol       Trigger              N/A                162     148     121     0.82    0.75    0.78
    Alcohol       Type                 N/A                 32      2       1      0.50    0.03    0.06
    ...
    Employment    History              N/A                 17      0       0      0.00    0.00    0.00
    Employment    StatusEmploy         employed            27      13      11     0.85    0.41    0.55
    Employment    StatusEmploy         none                0       2       0      0.00    0.00    0.00
    Employment    StatusEmploy         on_disability       6       0       0      0.00    0.00    0.00
    Employment    StatusEmploy         retired             34      1       1      1.00    0.03    0.06
    Employment    StatusEmploy         student             1       0       0      0.00    0.00    0.00
    Employment    StatusEmploy         unemployed          22      29      10     0.34    0.45    0.39
    Employment    Trigger              N/A                 90      57      42     0.74    0.47    0.57
    Employment    Type                 N/A                286      36      13     0.36    0.05    0.08
    ...


    '''

    arg_dict = vars(args)

    score_brat( \
        gold_dir = arg_dict['gold_dir'],
        predict_dir = arg_dict["predict_dir"],
        labeled_args = arg_dict["labeled_args"],
        score_trig = arg_dict["score_trig"],
        score_span = arg_dict["span_score"],
        score_labeled = arg_dict["span_labeled"],
        path = arg_dict["output"])



if __name__ == '__main__':

    parser = get_argparser()
    args = parser.parse_args()

    main(args)
