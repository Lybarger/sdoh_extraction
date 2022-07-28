
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil

import os
import pandas as pd

import brat_scoring.constants as C

from brat_scoring.scoring import PRF



def micro_average_subtypes(df):
    '''
    Calculate micro average across subtypes in event evaluation
        Input is dataframe
    '''

    # aggregate counts across subtype
    df = df.groupby([C.EVENT, C.ARGUMENT]).sum([C.NT, C.NP, C.TP])

    # reset index
    df = df.reset_index()

    # update precision, recall, and F measure calc
    df = PRF(df)

    # replace n/a with 0's
    df = df.fillna(0)

    return df


def micro_average_subtypes_csv(input_path, output_path=None, suffix='_micro'):
    '''
    Calculate micro average across subtypes in event evaluation
        Input is path to csv
    '''

    df = pd.read_csv(input_path)
    df = micro_average_subtypes(df)

    if output_path is None:
        fn, ext = os.path.splitext(input_path)
        output_path = f"{fn}{suffix}{ext}"

    assert input_path != output_path

    df.to_csv(output_path, index=False)

    return df



path = '/home/lybarger/sdoh_challenge/analyses/step112_multi_spert_infer/eval/sdoh_challenge_dev_uw_FAILED/scores_trig_exact_span_exact.csv'

df = micro_average_subtypes_csv(path)

print(df)
