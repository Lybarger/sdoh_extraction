

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.insert(0, '/home/lybarger/brat_scoring')

import argparse
from zipfile import ZipFile
from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
import shutil
import logging
import json
import config.paths as paths
import config.constants as C
from utils.path_utils import create_project_folder, define_logging
import re
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import math
import seaborn as sns

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.lines as mlines
from matplotlib.patches import Patch


from brat_scoring.corpus import Corpus
from brat_scoring.scoring import score_docs
from brat_scoring.constants import EXACT, LABEL, OVERLAP, PARTIAL, MIN_DIST, SPACY_MODEL, EVENT, OVERALL, F1, P, R, NT, NP, TP, ARGUMENT, TRIGGER, SUBTYPE
from brat_scoring.constants_sdoh import STATUS_TIME, TYPE_LIVING, STATUS_EMPLOY
from brat_scoring.constants_sdoh import ALCOHOL, DRUG, TOBACCO, EMPLOYMENT, LIVING_STATUS
from brat_scoring.constants_sdoh import LABELED_ARGUMENTS, SPAN_ONLY_ARGUMENTS, NONE

import brat_scoring.constants_sdoh as SC


from runs.step219_error_analysis_score import SUBSTANCE_TYPES, UNKOWN, SIMPLE, COMPLEX, CATEGORY, ID, SUBSET, SOURCE, SUBTASK
from runs.step219_error_analysis_score import get_prf

FONTNAME='serif'

SUBSTANCE_TYPES = [SC.ALCOHOL, SC.DRUG, SC.TOBACCO]

TITLE = 'title'

NOTES = 'notes'
CRITERIA = "criteria"
NAME = "name"
SELECTION = "selection"


SUB_RF = (SC.STATUS_TIME, [SC.NONE, SC.CURRENT, SC.PAST])

RISK_FACTORS = {}
RISK_FACTORS[SC.ALCOHOL] =  SUB_RF        
RISK_FACTORS[SC.DRUG] =     SUB_RF
RISK_FACTORS[SC.TOBACCO] =  SUB_RF
RISK_FACTORS[SC.EMPLOYMENT] =       (SC.STATUS_EMPLOY, [SC.EMPLOYED, SC.UNEMPLOYED, SC.RETIRED, SC.ON_DISABILITY])
RISK_FACTORS[SC.LIVING_STATUS] =    (SC.TYPE_LIVING,   [SC.WITH_FAMILY, SC.WITH_OTHERS, SC.ALONE, SC.HOMELESS])

MARKER_SIZE = 6


DPI = 600

Y2TICKS = 'y2ticks'
AVG_EVENTS_PER_NOTE = 'Avg. events/note'
# Y2LABEL = '# gold events'
Y2LABEL = AVG_EVENTS_PER_NOTE
TOTAL_EVENTS = 'Total events'
EVENT_COUNT = '# Events'


def get_scores_by_arg(df):


    groups = {'Trigger':[TRIGGER], 'Labeled arg.':LABELED_ARGUMENTS, 'Span-only arg.': SPAN_ONLY_ARGUMENTS}
    # groups = {'Trigger':[TRIGGER], 'Labeled arguments':LABELED_ARGUMENTS, 'Span-only arguments': SPAN_ONLY_ARGUMENTS}

    # Check for duplicates
    args = [v for k, V in groups.items() for v in V]
    assert len(args) == len(set(args))

    count_all = len(df)
    count_increment = 0
    rows = []
    for name, group in groups.items():

        df_temp = df[df[ARGUMENT].isin(group)]

        count_increment += len(df_temp)

        d = get_prf(df_temp, name=name)
        rows.append(d)

    assert count_all == count_increment

    df = pd.DataFrame(rows)

    return df



def font_adjust(font_size=10, font_family=FONTNAME, font_type='Times New Roman'):


    # plt.rc('text', usetex=True)

    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.serif'] = [font_type] + plt.rcParams['font.serif']

    params = {'axes.labelsize': font_size, 'axes.titlesize':font_size, 'legend.fontsize': font_size, 'xtick.labelsize': font_size, 'ytick.labelsize': font_size}
    mpl.rcParams.update(params)




def filter_df(df, criteria):
    df_ = df.copy()
    for column, value in criteria:
        df_ = df_[df_[column] == value]
    return df_

def plot_by_subtask(df_dict, path, config,\
    figsize = (7.25, 8.5),
    font_size = 10,
    dpi = DPI,
    y2label = AVG_EVENTS_PER_NOTE,
    y2step = 1,
    # y2ticks = None,
    ymargin = 1.15,
    y2margin = 1.15,
    line2color = 'black'
    ):

    font_adjust(font_size=font_size)

    n = len(config)

    fig, axs = plt.subplots( \
            nrows = n,
            ncols = 1,
            figsize = figsize)


    assert len(df_dict) == len(config)

    for i, (c, (name, df))  in enumerate(zip(config, df_dict.items())):


        criteria = c[CRITERIA]
        title = c[TITLE]
        note_count = c[NOTES]
        y2ticks = c[Y2TICKS]


        assert name == c[NAME]

        logging.info(f'{criteria} \n{df}')

        ax = axs[i]

        #set seaborn plotting aesthetics
        sns.set(style='white')

        #create grouped bar chart


        sns.barplot(x=EVENT, y=F1, hue='name', data=df,
                    palette='tab10', ax=ax)

        ax.set_title(title, fontweight="bold", size=font_size)
        ax.set_ylim(0, ymargin)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        ax.set(xlabel=None)


        font_adjust(font_size=font_size)

        ax2 = ax.twinx()
        # plt.setp(ax2.get_yticklabels(), visible=False)
        # ax2.tick_params(axis='both', which='both', length=0)

        print(df)
        print(note_count)


        df_temp = df[df[NAME] == TRIGGER]

        df_temp['Avg. Events'] = df_temp[NT]/note_count
        


        X = ax.get_xticks()
        Y = df_temp['Avg. Events']
        # Y[0] = None

        X = X[1:]
        Y = Y[1:]

        ax2.plot(X, Y, \
                linestyle='none', 
                mec=line2color, 
                color=line2color, 
                mfc=line2color, 
                ms=MARKER_SIZE, 
                marker='P', 
                label=AVG_EVENTS_PER_NOTE)


        ax2.set_ylabel(y2label, fontsize=font_size, fontname=FONTNAME)
        ax2.tick_params(labelsize=font_size)

        ax2.set_ylim(0, y2ticks[-1]*y2margin)
        ax2.set_yticks(y2ticks, fontsize=font_size, fontfamily=FONTNAME)


        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=font_size-2, fontfamily=FONTNAME)

        # xticklabels = ax.get_xticklabels()


        A = df_temp[EVENT]
        B = df_temp[NT]
        event_count_dict = OrderedDict(zip(A, B))
        xticklabels2 = []
        for lab, val in event_count_dict.items():
            xticklabels2.append(f'{lab}\n(n={val:,.0f})')
        
        ax.set_xticklabels(xticklabels2)




        ax.get_legend().remove()

        if i == n -1:

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()

            # ax.legend(lines + lines2, labels + labels2, loc=0)
            ax.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.5), ncols=4)

            # lines, labels = ax.get_legend_handles_labels()

            # ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.4), ncols=4)

    fig.tight_layout()
    # fig.savefig(path, dpi=dpi, bbox_inches='tight')
    fig.savefig(path, dpi=dpi)

    plt.close()

    return True


def plot_by_density(df_dict, path, \
    config,
    figsize = (7.25, 8.5),
    font_size = 10,
    dpi = DPI,
    y2label = TOTAL_EVENTS,
    xlabel = 'Event density (# gold events per note)',
    y2margin = 0.8,
    y2step = 500,
    # y2ticks = None,
    ymargin = 1.15,
    event_map={LIVING_STATUS:'Living Status'},
    ):


    font_adjust(font_size=font_size)

    n = len(df_dict)

    fig, axs = plt.subplots( \
            nrows = n,
            ncols = 1,
            figsize = figsize)

    for i, (name, df)  in enumerate(df_dict.items()):




        y2ticks = config[Y2TICKS]

        logging.info(f'{name} \n{df}')

        ax = axs[i]

        #set seaborn plotting aesthetics
        sns.set(style='white')

        #create grouped bar chart
        sns.barplot(x='Count', y=F1, hue='name', data=df,
                    palette='tab10', ax=ax)




        name = event_map.get(name, name)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set(xlabel=None)

       
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=font_size-2, fontfamily=FONTNAME)

        ax.set_ylim(0, ymargin)

        font_adjust(font_size=font_size)




        ax2 = ax.twinx()

        df_temp = df[df['name'] == 'Trigger']
        df_temp[NT] = df_temp[NT]

        title = f'{name} (n={df_temp[NT].sum():,.0f})'
        ax.set_title(title, fontweight="bold", size=font_size)

        sns.lineplot(data=df_temp, \
            x = 'Count', y=NT, linestyle='none', mec='black', 
            color='black', mfc='black', ms=MARKER_SIZE, marker='P', ax=ax2, label=TOTAL_EVENTS)

        ax2.set_ylabel(y2label, fontsize=font_size, fontname=FONTNAME)
        ax2.tick_params(labelsize=font_size)



        if y2ticks is None:
            m = df_temp[NT].max()
            m = math.ceil(m/y2margin)
            y2ticks = range(0, m+y2step, y2step)

        ax2.set_ylim(0, y2ticks[-1]*ymargin)
        ax2.set_yticks(y2ticks, fontsize=font_size, fontfamily=FONTNAME)

        ax2.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        # #ax2.yaxis.set_major_formatter('{x}K')

        labels = ax2.get_yticklabels()
        fontname = FONTNAME
        [label.set_fontname(fontname) for label in labels]





        font_adjust(font_size=font_size)

        ax.get_legend().remove()
        ax2.get_legend().remove()

    ax.set_xlabel(xlabel, fontweight='bold')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # ax.legend(lines + lines2, labels + labels2, loc=0)
    ax.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.5), ncols=4)


    fig.tight_layout()
    # fig.savefig(path, dpi=dpi, bbox_inches='tight')
    fig.savefig(path, dpi=dpi)

    plt.close()

    return True


def format_event_type(event_type):
    if event_type == LIVING_STATUS:
        return 'Living Status'
    if event_type == EMPLOYMENT:
        return 'Employ.'
    else:
        return event_type

def analysis_by_subtask(df, path, config,  event_types):

    analysis_dir = os.path.join(path, "subtask")
    create_project_folder(analysis_dir)

    print(df.head())

    logging.info(f"Count, all:          {len(df)}")


    df_counts = df.groupby([SOURCE, SUBSET, SELECTION])[ID].nunique()
    f = os.path.join(analysis_dir, f"counts.csv")
    df_counts.to_csv(f)

    df_dict = OrderedDict()
    for c in config:
        name = c[NAME]
        criteria = c[CRITERIA]

        note_count = c[NOTES]


        df_ = filter_df(df, criteria)

        logging.info("="*80)
        logging.info(f"Criteria:                 {criteria}")
        logging.info("="*80)        
        logging.info(f"Count, subtask/subset:   {len(df_)}")
        
        dfs = []


        df_scores = get_scores_by_arg(df_)
        for column, value in criteria:
            df_scores[column] = value

        df_scores.insert(0, EVENT, 'All')
        dfs.append(df_scores)
                
        for event_type in event_types:

            df_event = df_[df_[EVENT] == event_type]
            logging.info("")
            logging.info(f"Event:               {event_type}")
            logging.info(f"Count, event:        {len(df_event)}")        

            df_scores = get_scores_by_arg(df_event)

            
            for column, value in criteria:
                df_scores[column] = value

            event_type2 = format_event_type(event_type)

            df_scores.insert(0, EVENT, event_type2)

            dfs.append(df_scores)
            
        df_dict[name] = pd.concat(dfs)

        logging.info(f"\n{df_dict[name]}")


    for name, df_ in df_dict.items():
        f = os.path.join(analysis_dir, f"scores_{name}.csv")
        df_.to_csv(f)

    plot_by_subtask(df_dict, \
        path = os.path.join(analysis_dir, 'subtask_comparison.png'), 
        config = config)
    



def analysis_by_density(df, path, event_types, config, subtasks=['a', 'b', 'c'],
                ranges=[(1,2), (2,3), (3,20)]):


    analysis_dir = os.path.join(path, "density")
    create_project_folder(analysis_dir)

    print(df.head())

    logging.info(f"Count, all:          {len(df)}")

    df_dict = OrderedDict()


    for c in config:
        name = c[NAME]
        criteria = c[CRITERIA]


        df_config = filter_df(df, criteria)

        logging.info("="*80)
        logging.info(f"Criteria:                 {criteria}")
        logging.info("="*80)  

        for event_type in event_types:

            df_ = df_config[df_config[EVENT] == event_type]

            logging.info("-"*80)
            logging.info(f"Event:                   {event_type}")
            logging.info("-"*80)        
            logging.info(f"Count, event_type:       {len(df_)}")
            
            dfs = []

            count_all = len(df_)
            count_increment = 0
                
            for low, hi in ranges:

                df_count = df_[(df_[f'{event_type}_Count'] >= low) & (df_[f'{event_type}_Count'] < hi)]

                count_increment += len(df_count)

                if hi == low + 1:
                    count = str(low)
                else:
                    count = f'{low}+'

                logging.info("")
                logging.info(f"Count:                   {c}")
                logging.info(f"len:                     {len(df_count)}")        

                df_scores = get_scores_by_arg(df_count)


                df_scores[EVENT] = event_type
                df_scores['Count'] = count

                dfs.append(df_scores)


            df_dict[event_type] = pd.concat(dfs)

            logging.info(f"\n{df_dict[event_type]}")

        for event_type, df_ in df_dict.items():
            f = os.path.join(analysis_dir, f"scores_{name}_{event_type}.csv")
            df_.to_csv(f)

        plot_by_density(df_dict, \
            path = os.path.join(analysis_dir, f'density_{name}.png'), 
            config = c)
    


def plot_by_risk(df_dict, path, \
    figsize = (5.55, 7.5),
    font_size = 10,
    dpi = DPI,
    y2label = EVENT_COUNT,
    y2margin = 1.15,
    y2step = 1000,
    y2ticks = [0, 100, 200, 300],
    ymargin = 1.15,
    bar_color = 'b',
    y2color = 'black',
    y2marker = 'P'
    ):

    nbars = 0
    for i, (name, df)  in enumerate(df_dict.items()):
        nbars = max(nbars, len(df))

    font_adjust(font_size=font_size)

    n = len(df_dict)

    fig, axs = plt.subplots( \
            nrows = n,
            ncols = 1,
            figsize = figsize)

    for i, (name, df)  in enumerate(df_dict.items()):

        logging.info(f'{name} \n{df}')

        ax = axs[i]

        #set seaborn plotting aesthetics
        sns.set(style='white')

        #create grouped bar chart
        sns.barplot(x=SUBTYPE, y=F1, data=df,
                    color=bar_color, ax=ax)
        # ax.set(xlabel='x', ylabel=F1)

        # sns.barplot(x=SUBTYPE, y=F1, hue=NAME, data=df,
        #             palette='tab10', ax=ax)

        ax.set_title(name, fontweight="bold", size=font_size)

        ax.set_xlim(-0.5,nbars-0.5)

        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set(xlabel=None)

       
        X = []
        for j, container in enumerate(ax.containers):
            ax.bar_label(container, fmt='%.2f', fontsize=font_size-2, fontfamily=FONTNAME)

            # print(j, container.__dict__)
            for patch in container.patches:
                x, y = patch.get_xy()
                w = patch.get_width()
                X.append(x + w/2)

        ax2 = ax.twinx()
        # sns.barplot(x=SUBTYPE, y=NT, hue=NAME, data=df,
        #             palette='tab10', ax=ax2)

        

        Y = range(0, len(X))
        Y = df[EVENT_COUNT]
        
        ax2.plot(X, Y, \
            linestyle='none', 
            mec=y2color, 
            color=y2color, 
            mfc=y2color, 
            ms=MARKER_SIZE, 
            marker=y2marker, 
            label=EVENT_COUNT)

        #sns.lineplot(data=df, x=SUBTYPE, y=NT, linestyle='none', mec='black', color='black', mfc='black', ms=8, marker='o', ax=ax2, label=y2label)

        ax2.set_ylabel(y2label, fontsize=font_size, fontname=FONTNAME)
        ax2.tick_params(labelsize=font_size)

    #     if y2ticks is None:
    #         m = df_temp[NT].max()
    #         m = math.ceil(m/y2margin)
    #         y2ticks = range(0, m+y2step, y2step)

        ax2.set_ylim(0, y2ticks[-1]*y2margin)
        ax2.set_yticks(y2ticks, fontsize=font_size, fontfamily=FONTNAME)

    #     # #ax2.yaxis.set_major_formatter('{x}K')

        labels = ax2.get_yticklabels()
        fontname = FONTNAME
        [label.set_fontname(fontname) for label in labels]


        ax.set_ylim(0, ymargin)

        font_adjust(font_size=font_size)

        #ax.get_legend().remove()
        # ax2.get_legend().remove()


    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # # ax.legend(lines + lines2, labels + labels2, loc=0)
    #ax.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.4), ncols=5)
    # ax.legend([lines[0]], ['F1'])


    lines = [Patch(facecolor=bar_color, edgecolor=bar_color, label=F1),
            mlines.Line2D([], [], color=y2color, marker=y2marker, linestyle='None', markersize=MARKER_SIZE, label=y2label) ]
    ax.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, -0.4), ncols=2)






    fig.tight_layout()
    # fig.savefig(path, dpi=dpi, bbox_inches='tight')
    fig.savefig(path, dpi=dpi)

    plt.close()

    return True



def analysis_by_risk_factor(df, path, risk_factors=RISK_FACTORS):


    event_map={LIVING_STATUS:'Living Status'}
    argument_map={STATUS_TIME: 'Status Time', STATUS_EMPLOY: 'Status Employ', TYPE_LIVING: 'Type Living'}

    analysis_dir = os.path.join(path, "risk_factors")
    create_project_folder(analysis_dir)

    print(df.head())

    logging.info(f"Count, all:          {len(df)}")

    df_dict = OrderedDict()


    df_dict = {}

    for event_type, (argument, subtypes) in risk_factors.items():

        df_event = df[df[EVENT] == event_type]

        rows = []

        for st in subtypes:

            df_subtype = df_event[(df_event[ARGUMENT] == argument) & (df_event[SUBTYPE] == st)]

            assert list(df_subtype[EVENT].unique())[0] == event_type
            assert list(df_subtype[ARGUMENT].unique())[0] == argument
            assert list(df_subtype[SUBTYPE].unique())[0] == st

            d = {} 
            d[EVENT] = event_type
            d[ARGUMENT] = argument
            d[SUBTYPE] = st.replace('_', ' ')
            d.update(get_prf(df_subtype))
            d[EVENT_COUNT] = d[NT]
            rows.append(d)

        df_rows = pd.DataFrame(rows)
        f = os.path.join(analysis_dir, f"scores_risk_factors_{event_type}.csv")
        df_rows.to_csv(f)


        event_type2 = event_map.get(event_type, event_type)
        argument2 = argument_map.get(argument, argument)

        df_dict[f'{event_type2} - {argument2}'] = df_rows


    f = os.path.join(analysis_dir, 'risk_factor.png')
    plot_by_risk(df_dict, path=f)


def main(args):








    y2ticks = [0.00, 0.25, 0.50, 0.75, 1.0, 1.25]
    # config_by_subtask = []
    # config_by_subtask.append({NAME: 'a',               CRITERIA: [(SUBTASK, "a")],                         TITLE: r'A ($\mathcal{D}_{test}^{mimic}$)', NOTES:373,  Y2TICKS: y2ticks})
    # config_by_subtask.append({NAME: 'b_train',         CRITERIA: [(SUBTASK, "b"), (SUBSET, 'train')],      TITLE: r'B ($\mathcal{D}_{train}^{uw}$)',   NOTES:1751, Y2TICKS: y2ticks})
    # config_by_subtask.append({NAME: 'b_dev',           CRITERIA: [(SUBTASK, "b"), (SUBSET, 'dev')],        TITLE: r'B ($\mathcal{D}_{dev}^{uw}$)',     NOTES:259,  Y2TICKS: y2ticks})
    # config_by_subtask.append({NAME: 'c',               CRITERIA: [(SUBTASK, "c")],                         TITLE: r'C ($\mathcal{D}_{test}^{uw}$)',    NOTES:518,  Y2TICKS: y2ticks})    

    # config_count = []
    # config_count.append({NAME: 'a',         TITLE: r'A ($\mathcal{D}_{test}^{mimic}$)',  Y2TICKS: range(0,500,100),   CRITERIA: [(SUBTASK, "a")],                        NOTES:373})
    # config_count.append({NAME: 'b_train',   TITLE: r'B ($\mathcal{D}_{train}^{uw}$)',    Y2TICKS: range(0,2500,500),  CRITERIA: [(SUBTASK, "b"), (SUBSET, 'train')],     NOTES:1751})
    # config_count.append({NAME: 'b_dev',     TITLE: r'B ($\mathcal{D}_{dev}^{uw}$)',      Y2TICKS: range(0,300,50),   CRITERIA: [(SUBTASK, "b"), (SUBSET, 'dev')],       NOTES:259})
    # config_count.append({NAME: 'c',         TITLE: r'C ($\mathcal{D}_{test}^{uw}$)',     Y2TICKS: range(0,600,100),  CRITERIA: [(SUBTASK, "c")],                        NOTES:518})     


    # config_risk = []
    # config_risk.append({NAME: r'A ($\mathcal{D}_{test}^{mimic}$)',  CRITERIA: [(SUBTASK, "a")],                        NOTES:373})
    # config_risk.append({NAME: r'B ($\mathcal{D}_{train}^{uw}$)',    CRITERIA: [(SUBTASK, "b"), (SUBSET, 'train')],     NOTES:1751})
    # config_risk.append({NAME: r'B ($\mathcal{D}_{dev}^{uw}$)',      CRITERIA: [(SUBTASK, "b"), (SUBSET, 'dev')],       NOTES:259})
    # config_risk.append({NAME: r'C ($\mathcal{D}_{test}^{uw}$)',     CRITERIA: [(SUBTASK, "c")],                        NOTES:518})    


    # config_by_subtask.append({SUBTASK: 'b', SUBSET: None,     TITLE: r'Subtask B ($\mathcal{D}^{uw}$)'})
    event_types = [ALCOHOL, DRUG, TOBACCO, EMPLOYMENT, LIVING_STATUS]


    source = '/home/lybarger/sdoh_challenge/analyses/step219_error_analysis_score/scores.csv'

    destination = paths.error_analysis_figures

    create_project_folder(destination)

    define_logging(destination)

    logging.info("")
    logging.info('='*80)
    logging.info('step220_error_analysis_figures')
    logging.info('='*80)

    logging.info(f'destination:\t{destination}')

    df = pd.read_csv(source)

    f = os.path.join(destination, f"data.csv")
    df.to_csv(f)


    # analysis_by_density(df, \
    #         path = destination, 
    #         event_types = event_types,
    #         config = config_count)


    # analysis_by_subtask(df, \
    #         path = destination, 
    #         config = config_by_subtask,
    #         event_types = event_types)


    analysis_by_risk_factor(df, path=destination)




if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(add_help=False)
    args, _ = arg_parser.parse_known_args()

    sys.exit(main(args))  # next section explains the use of sys.exit
