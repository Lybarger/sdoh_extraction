from collections import OrderedDict
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scoring.scoring import summarize_event_df, summarize_event_csvs
import config.constants as C
from scoring.scoring import score_brat


brat_true = '/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/e10_d2/brat_true/'
brat_predict = '/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/e10_d2/brat_predict/'
destination = '/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/e10_d2/temp/'

f1 = '/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/e10_d2/temp/scores_strict.csv'
f2 = '/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/e10_d2/temp/scores_relaxed_trig_overlap.csv'
f3 = '/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/e10_d2/temp/scores_relaxed_trig_min_dist.csv'
f4 = '/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/e10_d2/temp/scores_relaxed_all.csv'
f5 = '/home/lybarger/sdoh_challenge/analyses/step110_extraction/train/e10_d2/temp/scores_relaxed_partial.csv'
labeled_args = [C.STATUS_TIME, C.TYPE_LIVING, C.STATUS_EMPLOY]

scoring_defs = [ \
                dict( \
                    score_trig = C.EXACT,
                    score_span = C.EXACT,
                    score_labeled = C.LABEL,
                    description = 'strict'),
                dict( \
                    score_trig = C.OVERLAP,
                    score_span = C.EXACT,
                    score_labeled = C.LABEL,
                    description = 'relaxed_trig_overlap'),
                dict( \
                    score_trig = C.MIN_DIST,
                    score_span = C.EXACT,
                    score_labeled = C.LABEL,
                    description = 'relaxed_trig_min_dist'),
                dict( \
                    score_trig = C.MIN_DIST,
                    score_span = C.OVERLAP,
                    score_labeled = C.LABEL,
                    description = 'relaxed_all'),
                dict( \
                    score_trig = C.MIN_DIST,
                    score_span = C.PARTIAL,
                    score_labeled = C.LABEL,
                    description = 'relaxed_partial'),
]

for scoring_def in scoring_defs:
    df = score_brat(brat_true, brat_predict, \
                                labeled_args = labeled_args,
                                path = destination,
                                **scoring_def)



d = OrderedDict([('strict', f1), ('relaxed_trig_overlap', f2), ('relaxed_trig_min_dist', f3), ('relaxed_all', f4), ('relaxed_partial', f5)])


df = summarize_event_csvs(d)
print(df)
