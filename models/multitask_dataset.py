

from collections import OrderedDict, Counter

import torch
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from transformers import AutoTokenizer, AutoModel, AutoConfig
#torch.multiprocessing.set_start_method("spawn")

import pandas as pd

import sys
import os
import errno
from datetime import datetime
from tqdm import tqdm
import numpy as np
import logging
from tqdm import tqdm
import joblib
import math
import json


from models.utils import pad_embedding_seq
from models.utils import pad_sequences

# from utils.seq_prep import preprocess_tokens_doc
from models.utils import create_mask,  map_2D
from models.crf import  BIO_to_span

# from corpus.event import Event, Span, events2sent_labs, events2seq_tags
from models.utils import map_dict_builder
from models.xfmrs import tokens2wordpiece, get_embeddings, embed_len_check

import config.constants as C

START_TOKEN = '<c>'
END_TOKEN = '<e>'

def get_label_map(label_def):


    # Initialize output
    to_id = OrderedDict()
    to_lab = OrderedDict()
    num_tags = OrderedDict()

    # Loop on label categories
    for name, lab_def in label_def.items():

        to_id[name] = OrderedDict()
        to_lab[name] = OrderedDict()
        num_tags[name] = OrderedDict()

        # Loop on argument types in label map
        for evt_typ, map_ in lab_def[C.LAB_MAP].items():

            # Account for BIO prefixes
            if lab_def[C.LAB_TYPE] == C.SEQ:
                map_ = [(C.OUTSIDE, map_[0])] + \
                       [(p, m) for m in map_[1:] for p in [C.BEGIN, C.INSIDE]]

            # Generate map
            to_id[name][evt_typ], to_lab[name][evt_typ] = \
                                                  map_dict_builder(map_)
            num_tags[name][evt_typ] = len(map_)

    logging.info("Label to ID mapping functions:")
    for name, arguments in to_id.items():
        logging.info('\t{}'.format(name))
        for evt_typ, map_ in arguments.items():
            logging.info('\t\t{}'.format(evt_typ))
            for k, v in map_.items():
                logging.info('\t\t\t{} --> {}'.format(k, v))

    logging.info("ID to Label mapping functions:")
    for name, arguments in to_lab.items():
        logging.info('\t{}'.format(name))
        for evt_typ, map_ in arguments.items():
            logging.info('\t\t{}'.format(evt_typ))
            for k, v in map_.items():
                logging.info('\t\t\t{} --> {}'.format(k, v))

    logging.info('')
    logging.info('Number of tags:')
    for name, lab_def in num_tags.items():
        logging.info('\t{}'.format(name))
        for name, cnt in lab_def.items():
            logging.info('\t\t{} = {}'.format(name, cnt))


    return (to_id, to_lab, num_tags)



def events2multitask_labs(events, label_def, max_len, pad_start, \
                               label_to_id=None, include_prefix=True):
    '''
    Get multitask labels from events

    Parameters
    ----------
    events: document labels as sequence of sentences as sequence of Event
            e.g. [[Event(), ... Event()],
                  [Event(), ... Event()],
                  [Event(), ... Event()]]

    label_map: nested event type-argument type combinations as dict of list
            e.g. {'Trigger:
                    {'Alcohol': ['Outside', 'Alcohol']},...
                 {'Status:
                    {'Alcohol': ['Outside', 'none',...]},...
                 {'Entity':
                    {'Alcohol': ['Amount', 'Frequency']},...
                    }

    max_len: maximum sequence length as int

    pad_start: Boolean indicating whether start of sequence is padded
                with start of sequence token


    Returns
    -------
    trig:
    status:
    entity:
    prefixes:

    '''

    # Sentence count
    n = len(events)

    # Initialize output
    y = [OrderedDict() for _ in range(n)]

    # Iterate over label types
    for name, lab_def in label_def.items():

        lab2id = None if label_to_id is None else label_to_id[name]

        # Sentence-level label
        if lab_def[LAB_TYPE] == SENT:
            y_tmp = events2sent_labs( \
                        events = events,
                        label_map = lab_def[LAB_MAP],
                        arg_type = lab_def[ARGUMENT],
                        label_to_id = lab2id)

        # Token-level labels
        elif lab_def[LAB_TYPE] == SEQ:
            y_tmp = events2seq_tags( \
                        events = events,
                        label_map = lab_def[LAB_MAP],
                        max_len = max_len,
                        pad_start = pad_start,
                        include_prefix = include_prefix,
                        label_to_id = lab2id)

        else:
            raise ValueError("Invalid label type:\t{}".format(lab_def[LAB_TYPE]))


        assert len(y_tmp) == len(y)
        for i, y_ in enumerate(y_tmp):
            y[i][name] = y_

    return y


def sent2multitask_labs(sentence, label_def, max_len, pad_start, \
                               label_to_id=None, include_prefix=True):
    '''
    Get multitask labels from events

    Parameters
    ----------
    events: document labels as sequence of sentences as sequence of Event
            e.g. [[Event(), ... Event()],
                  [Event(), ... Event()],
                  [Event(), ... Event()]]

    label_map: nested event type-argument type combinations as dict of list
            e.g. {'Trigger:
                    {'Alcohol': ['Outside', 'Alcohol']},...
                 {'Status:
                    {'Alcohol': ['Outside', 'none',...]},...
                 {'Entity':
                    {'Alcohol': ['Amount', 'Frequency']},...
                    }

    max_len: maximum sequence length as int

    pad_start: Boolean indicating whether start of sequence is padded
                with start of sequence token


    Returns
    -------
    trig:
    status:
    entity:
    prefixes:

    '''



    tokens = sentence["tokens"]
    entities = sentence["entities"]
    subtypes = sentence["subtypes"]
    relations = sentence["relations"]

    '''
    Initialize output to negative labels
    '''
    # iterate over label types (trigger, status, type, entity)
    y = OrderedDict()
    for name, lab_def in label_def.items():

        lab_type = lab_def[C.LAB_TYPE]

        # iterate over event types (alcohol, drag, tobacco, etc)
        y[name] = OrderedDict()
        for event_type, label_set in lab_def[C.LAB_MAP].items():

            neg_lab = label_set[0]

            lab2id = label_to_id[name][event_type]

            # Sentence-level label
            if lab_type == C.SENT:

                # Initialize to negative label
                y[name][event_type] = lab2id[neg_lab]

            # Token-level labels
            elif lab_type == C.SEQ:

                if include_prefix:
                    labs = [lab2id[(C.OUTSIDE, neg_lab)] for _ in range(max_len)]

                else:
                    labs = [lab2id[neg_lab] for _ in range(max_len)]

                y[name][event_type] = torch.LongTensor(labs)


    '''
    Decode entities that are triggers
    '''
    for e in entities:

        argument_type = e["type"]
        if argument_type in label_def[C.TRIGGER][C.LAB_MAP]:
            lab2id = label_to_id[C.TRIGGER][event_type]
            y[C.TRIGGER][argument_type] = lab2id[event_type]


    '''
    Decode relations
    '''
    for r in relations:

        # Always trigger, sentence level label
        head_index = r["head"]
        head_entity = entities[head_index]
        event_type = head_entity["type"]

        # lab2id = label_to_id[C.TRIGGER][event_type]
        # y[C.TRIGGER][event_type] = lab2id[event_type]


        # Always span-only argument sequence
        tail_index = r["tail"]
        tail_entity = entities[tail_index]
        argument_type = tail_entity["type"]
        argument_start = tail_entity["start"]
        argument_end = tail_entity["end"]


        lab2id = label_to_id[C.ENTITY][event_type]


        for i in range(argument_start, argument_end):
            if i == 0:
                lab = lab2id[(C.BEGIN, argument_type)]
            else:
                lab = lab2id[(C.INSIDE, argument_type)]

            j = i + int(pad_start)
            if j < max_len:
                y[C.ENTITY][event_type][j] = lab

    '''
    Decode subtypes
    '''
    assert len(entities) == len(subtypes)
    for e, s in zip(entities, subtypes):

        event_type = e["type"]
        argument_start = s["start"]
        argument_end = s["end"]


        for argument_type, argument_label in s["type"].items():



            if argument_label != C.SUBTYPE_DEFAULT:



                if (event_type in y[argument_type]) and \
                   (event_type in label_to_id[argument_type]) and \
                   (argument_label in label_to_id[argument_type][event_type]):

                    id_exist = y[argument_type][event_type]
                    id_new = label_to_id[argument_type][event_type][argument_label]

                    y[argument_type][event_type] = max(id_exist, id_new)

    return y


def preprocess_X( \
                sentences,
                pretrained_path = None,
                tokenizer_path = None,
                device = None,
                max_len = 30,
                num_workers = 6,
                get_last = True,
                batch_size = 50,
                ):

    '''
    Preprocess tokenized input text

    Parameters
    ----------
    X: list of tokenized sentences,
            e.g.[['Patient', 'denies', 'tobacco', '.'], [...]]

    '''


    logging.info('='*72)
    logging.info('Preprocessing X')
    logging.info('='*72)

    # Flatten documents into sequence of sentences
    logging.info('Sentence count:\t{}'.format(len(sentences)))

    logging.info('Encoding input using transformer...')


    tokens = [s['tokens'] for s in sentences]


    # Get word pieces
    wp_toks, wp_ids, tok_idx = tokens2wordpiece( \
                            tokens = tokens,
                            tokenizer_path = tokenizer_path,
                            get_last = get_last)

    # Get sequence length, with start and and padding
    seq_lengths = [len(x) for x in tok_idx]


    # X as embedding
    embed = get_embeddings( \
                            word_piece_ids = wp_ids,
                            tok_idx = tok_idx,
                            pretrained_path = pretrained_path,
                            num_workers = num_workers,
                            batch_size = batch_size,
                            device = device)
    # Check lengths
    embed_len_check(tokens, embed)

    # Pad sequences of embedding
    embed = [pad_embedding_seq(x, max_len) for x in embed]

    # Convert embeddings to tensor
    embed = torch.tensor(embed)

    logging.info('Sequence length min:\t{}'.format(min(seq_lengths)))
    logging.info('Sequence length max:\t{}'.format(max(seq_lengths)))
    logging.info('Embedding dimensions:\t{}'.format(embed.shape))
    # logging.info('Embedding memory size:\t{}'.format(mem_size(embed)))



    logging.info('')

    return (tokens, embed, seq_lengths)


def preprocess_X_w2v(X, \
                embed_map = None,
                embed_matrix = None,
                max_len = 30,
                pad_start = True,
                pad_end = True,
                start_token = START_TOKEN,
                end_token = END_TOKEN):


    '''
    Preprocess tokenized input text

    Parameters
    ----------
    X: list of tokenized sentences,
            e.g.[['Patient', 'denies', 'tobacco', '.'], [...]]

    '''


    logging.info('='*72)
    logging.info('Preprocessing X, w2v')
    logging.info('='*72)

    # Flatten documents into sequence of sentences
    X = [sent for doc in X for sent in doc]
    logging.info('\tDocument count:\t{}'.format(len(X)))
    logging.info('\tFlattened documents to sequence of sentences')
    logging.info('\tSentence count:\t{}'.format(len(X)))

    # Include start and end of sequence padding tokens
    # (still variable length)
    tokens_proc = preprocess_tokens_doc(X, \
                    pad_start = pad_start,
                    pad_end = pad_end,
                    start_token = start_token,
                    end_token = end_token,
                    )

    # Get sequence length, with start and and padding
    seq_lengths = [len(x) for x in tokens_proc]

    # Map to token IDs
    ids = map_2D(tokens_proc, embed_map)

    # Pad so fixed length sequence
    ids = pad_sequences(ids, max_len)
    ids = torch.LongTensor(ids).cpu()
    embed_matrix = embed_matrix.cpu()
    #ids = torch.LongTensor(ids).to(embed_matrix.weight.device)

    embed = embed_matrix(ids)

    logging.info('')
    return (X, embed, seq_lengths)



def preprocess_y(sentences, label_def, max_len, pad_start, label_to_id):
    '''
    Convert events to multitask label IDs
    '''

    # Bail if no input
    if sentences is None:
        return None




    y = []

    for sent in sentences:
        y_sent = sent2multitask_labs( \
                                sentence = sent,
                                label_def = label_def,
                                max_len = max_len,
                                pad_start = pad_start,
                                label_to_id = label_to_id,
                                include_prefix = True)
        y.append(y_sent)

    return y





def sent_lab_span(labels, tokens, pad_start, id_to_label, type_, sent_idx):


    # Sequence length
    n = len(tokens)

    # Iterate over labels in sequence
    for i, lab in enumerate(labels):

        # Find first non-negative label
        if lab > 0:

            # Position of trigger start
            start = i - int(pad_start)
            start = min(max(start, 0), n-1)
            end = start + 1

            # Create span
            span = Span( \
                            type_ = type_,
                            sent_idx = sent_idx,
                            tok_idxs = (start, end),
                            tokens = tokens[start:end],
                            label = id_to_label[lab])
            return span

    # Return None, if nothing detected
    return None


def sent_lab_to_entity(labels, tokens, pad_start, id_to_label):


    # Sequence length
    n = len(tokens)

    # Iterate over labels in sequence
    for i, lab in enumerate(labels):

        # Find first non-negative label
        if lab > 0:

            # Position of trigger start
            start = i - int(pad_start)
            start = min(max(start, 0), n-1)
            end = start + 1

            # Create span
            entity = dict(type=id_to_label[lab], start=start, end=end)

            return entity

    # Return None, if nothing detected
    return None


def seq_tag_spans(labels, tokens, pad_start, id_to_label, sent_idx):


    # Get tags and span indices from BIO
    tag_start_end = BIO_to_span(labels, id_to_label)

    # Loop on spans
    spans = []
    for type_, start, end in tag_start_end:

        # Decrement token indices, if start padded
        start -= int(pad_start)
        start = max(start, 0)
        end -= int(pad_start)
        end = max(end, 0)

        spn = Span( \
                        type_ = type_,
                        sent_idx = sent_idx,
                        tok_idxs = (start, end),
                        tokens =  tokens[start:end],
                        label = None)
        spans.append(spn)

    return spans

def seq_tag_entities(labels, pad_start, id_to_label):


    # Get tags and span indices from BIO
    tag_start_end = BIO_to_span(labels, id_to_label)

    # Loop on spans
    entities = []
    for type_, start, end in tag_start_end:

        # Decrement token indices, if start padded
        start -= int(pad_start)
        start = max(start, 0)
        end -= int(pad_start)
        end = max(end, 0)

        print("type_", type_)

        entity = dict(type=type_, start=start, end=end)
        entities.append(entity)

    return entities


def decode_(y, tokens, id_to_label, pad_start, label_def):
    '''
    Postprocess predictions
    '''


    event_types = list(label_def[C.TRIGGER][C.LAB_MAP])

    assert len(y) == len(tokens)
    sentences = []
    for i, (y_, toks) in enumerate(zip(y, tokens)):
        sentence = {}
        sentence["tokens"] = toks

        print()

        print(toks)



        trigger_dict = {event_type:None for event_type in event_types}
        entity_dict =  {event_type:[] for   event_type in event_types}
        subtype_dict = {event_type:{} for   event_type in event_types}

        for name, labels_by_event in y_.items():
            print(name)

            # Label type (e.g. 'sentence' or 'sequence')
            label_type = label_def[name][C.LAB_TYPE]

            for event_type, labels in labels_by_event.items():
                print(event_type, labels)

                id2lab = id_to_label[name][event_type]

                if label_type == C.SENT:
                    entity = sent_lab_to_entity( \
                                    labels = labels,
                                    tokens = toks,
                                    pad_start = pad_start,
                                    id_to_label = id2lab,
                                    )
                    if entity is not None:
                        if name == C.TRIGGER:
                            trigger_dict[event_type] = entity
                        else:
                            subtype_dict[event_type][name] = entity

                elif label_type == C.SEQ:

                    entity_dict[event_type] = seq_tag_entities(labels, pad_start, id2lab)

                else:
                    raise ValueError(f"Invalid label type: \t {label}")

        print('trigger', trigger_dict)
        print('subtype', subtype_dict)
        print('entity ', entity_dict)

        X = START_HERE
        #
        # # Loop on label types (e.g. Trigger, Status, etc.)
        # for name, evt_labs in y_.items():
        #
        #     # Label type (e.g. 'sentence' or 'sequence')
        #     lab_typ = label_def[name][LAB_TYPE]
        #
        #     # Loop on event types for label type (e.g. Alcohol, Drug, etc.)
        #     for evt_typ, labs in evt_labs.items():
        #
        #         id2lab = id_to_label[name][evt_typ]
        #
        #         if lab_typ == SENT:
        #             span = sent_lab_span( \
        #                             labels = labs,
        #                             tokens = toks,
        #                             pad_start = pad_start,
        #                             id_to_label = id2lab,
        #                             type_ = name,
        #                             sent_idx = i,
        #                             )
        #
        #             if span is not None:
        #                 evt_spans[evt_typ].append(span)
        #
        #         elif lab_typ == SEQ:
        #             spans = seq_tag_spans( \
        #                             labels = labs,
        #                             tokens = toks,
        #                             pad_start = pad_start,
        #                             id_to_label = id2lab,
        #                             sent_idx = i,
        #                             )
        #             evt_spans[evt_typ].extend(spans)
        #         else:
        #             raise ValueError("invalid label type:\t{}".format(lab_typ))



    a = START_HERE

    # All possible event types
    evt_types = list(set([evt_typ for name, evt_labs in y[0].items() \
                                             for evt_typ in evt_labs]))
    # Loop on sentences
    assert len(y) == len(tokens)
    events_doc = []
    for i, (y_, toks) in enumerate(zip(y, tokens)):

        # Spans by event
        evt_spans = OrderedDict([(evt_typ, []) for evt_typ in evt_types])

        # Loop on label types (e.g. Trigger, Status, etc.)
        for name, evt_labs in y_.items():

            # Label type (e.g. 'sentence' or 'sequence')
            lab_typ = label_def[name][LAB_TYPE]

            # Loop on event types for label type (e.g. Alcohol, Drug, etc.)
            for evt_typ, labs in evt_labs.items():

                id2lab = id_to_label[name][evt_typ]

                if lab_typ == SENT:
                    span = sent_lab_span( \
                                    labels = labs,
                                    tokens = toks,
                                    pad_start = pad_start,
                                    id_to_label = id2lab,
                                    type_ = name,
                                    sent_idx = i,
                                    )

                    if span is not None:
                        evt_spans[evt_typ].append(span)

                elif lab_typ == SEQ:
                    spans = seq_tag_spans( \
                                    labels = labs,
                                    tokens = toks,
                                    pad_start = pad_start,
                                    id_to_label = id2lab,
                                    sent_idx = i,
                                    )
                    evt_spans[evt_typ].extend(spans)
                else:
                    raise ValueError("invalid label type:\t{}".format(lab_typ))

        # Convert spans into events
        events_sent = []
        for evt_typ, spans in evt_spans.items():

            # Build event
            if len(spans) > 0:
                evt = Event( \
                        type_ = evt_typ,
                        arguments = spans)
                events_sent.append(evt)

        events_doc.append(events_sent)

    return events_doc

def list_to_counts(X):

    counts = Counter(X)
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df = df.rename(columns={'index':'event', 0:'count'})
    return df

class MultitaskDataset(Dataset):
    """
    """
    def __init__(self, \
        dataset_path,
        label_def,
        pretrained_path,
        tokenizer_path,
        device,
        max_len = 30,
        num_workers = 6,
        get_last = True,
        batch_size = 50,
        mode = 'fit',
        pad_start = True
        ):
        super(MultitaskDataset, self).__init__()

        '''
        Parameters
        ----------
        X: list of tokenized documents,
                e.g. [doc [sent [tokens]]], [[['Patient', 'denies', 'tobacco', '.'], [...]]]
        y: list of documment events
                e.g. [doc [sent [events]]], [[[Event1, Event2, ...], [...]]]


        Returns
        -------

        '''

        self.dataset_path = dataset_path
        self.label_def = label_def
        self.pretrained_path = pretrained_path
        self.tokenizer_path = tokenizer_path
        self.device = device

        self.max_len = max_len
        self.num_workers = num_workers
        self.get_last = get_last
        self.batch_size = batch_size
        self.mode = mode
        self.pad_start = pad_start


        with open(dataset_path, "r") as f:
            self.sentences = json.load(f)

        self.sent_count = len(self.sentences)

        # Map to embeddings, and get sequence lengths
        self.tokens, self.X, self.seq_lengths = preprocess_X( \
                sentences = self.sentences,
                pretrained_path = self.pretrained_path,
                tokenizer_path = self.tokenizer_path,
                device = self.device,
                max_len = self.max_len,
                num_workers = self.num_workers,
                get_last = self.get_last,
                batch_size = self.batch_size
                )


        # else:
        #     self.tokens, self.X, self.seq_lengths = preprocess_X_w2v( \
        #             X = X,
        #             embed_map = word_embed_map,
        #             embed_matrix = word_embed_matrix,
        #             max_len = self.max_len,
        #             pad_start = pad_start,
        #             pad_end = pad_end)

        assert len(self.X) == self.sent_count

        self.mask = create_mask(self.seq_lengths, self.max_len)

        # Get label maps
        self.label_to_id, self.id_to_label, self.num_tags = \
                                          get_label_map(self.label_def)


        if self.mode == "fit":
            # Process labeled input
            self.y = preprocess_y( \
                                        sentences = self.sentences,
                                        label_def = self.label_def,
                                        max_len = self.max_len,
                                        pad_start = self.pad_start,
                                        label_to_id = self.label_to_id)
        else:
            self.y = None

        logging.info("")
        logging.info("Multitask Data set")
        logging.info("Sentence count:\t{}".format(self.sent_count))



    def __len__(self):
        return self.sent_count

    def __getitem__(self, index):

        # Current input and mask
        X = self.X[index]
        mask = self.mask[index]

        #Prediction (input only)
        if self.y is None:
            return (X, mask)

        # Supervised learning (input in labels)
        else:
            y = self.y[index]

            return (X, mask, y)



    def decode_(self, y):


        # iterate over sentences
        # pos_lab = []
        # for sentence in y:
        #
        #
        #     for name, lab_def in sentence.items():
        #         for event_type, labels in lab_def.items():
        #             if sum(labels) > 0:
        #                 pos_lab.append((K, k))
        # pos_lab = list_to_counts(pos_lab)
        # logging.info('Label predictions for decoding. Positive counts:\n{}'.format(pos_lab))


        events = decode_( \
                            y = y,
                            tokens = self.tokens,
                            id_to_label = self.id_to_label,
                            pad_start = self.pad_start,
                            label_def = self.label_def)

        event_counts1 = sum([len(sent) for sent in events])

        get_pos = lambda X: [k for x in X for k, v in x.items() for v_ in v if v_ > 0]



        logging.info("Multitask data set, decoding")
        logging.info("Event counts, by sent:\t{}".format(event_counts1))

        # Initialize output
        events_by_doc = [[[] for _ in range(c)] \
                                              for c in self.sent_counts]

        # Loop on documents
        i_sent = 0
        for j_doc, cnt in enumerate(self.sent_counts):

            tmp = events[i_sent:i_sent + cnt]
            for j_sent, sent in enumerate(tmp):
                for evt in sent:
                    for span in evt.arguments:
                        span.sent_idx = j_sent
            events_by_doc[j_doc] = tmp

            i_sent += cnt


        event_counts2 = sum([len(sent) for doc in events_by_doc for sent in doc])
        assert event_counts1 == event_counts2, '{} vs {}'.format(event_counts1, event_counts2)

        z = HERE_HERE

        return events_by_doc
