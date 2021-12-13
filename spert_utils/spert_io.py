from collections import Counter, OrderedDict
import re
import logging
import pandas as pd
import json
import os
import copy
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from corpus.labels import Event, Entity
from corpus.corpus_predict import CorpusPredict
from corpus.document_predict import DocumentPredict

from config.constants import SUBTYPE_DEFAULT

from corpus.tokenization import get_tokenizer
from corpus.brat import Attribute, Textbound, Event, get_unique_arg



ID = "id"
TOKENS = "tokens"
OFFSETS = "offsets"
ENTITIES = "entities"
RELATIONS = "relations"
SUBTYPES = "subtypes"
TYPE = "type"
START = "start"
END = "end"
HEAD = "head"
TAIL = "tail"
SENT_INDEX = "sent_index"
SENT_LABELS = "sent_labels"
DOC_TEXT = "doc_text"
VALUE = "value"


RELATION_DEFAULT = 'relation'
CHAR_COUNT = 12


FIELDS = [ID, DOC_TEXT, SENT_INDEX, OFFSETS]

def create_entity(type_, start, end):
    return {TYPE:type_, START: start, END: end}


def create_relation(head, tail, relation=RELATION_DEFAULT):
    return {TYPE: relation, HEAD: head, TAIL: tail}



def doc2spert(doc, event_types=None, entity_types=None, \
            skip_duplicate_spans=True, include_doc_text=False):


    sent_count = len(doc.tokens)

    events = doc.events( \
                    by_sent = True,
                    event_types = event_types,
                    entity_types = entity_types)

    # map events to entities
    spans = [set([]) for _ in range(sent_count)]
    entities = [[] for _ in range(sent_count)]
    subtypes = [[] for _ in range(sent_count)]

    entity_counter = Counter()
    for event in events:

        for i, argument in enumerate(event.arguments):


            tok_check = doc.tokens[argument.sent_index][argument.token_start:argument.token_end]
            assert tok_check == argument.tokens

            span = (argument.token_start, argument.token_end)
            entity =  (argument.type_,   argument.token_start, argument.token_end)

            st = argument.subtype
            if st is None:
                st = SUBTYPE_DEFAULT

            subtype = (st, argument.token_start, argument.token_end)

            if entity in entities[argument.sent_index]:
                pass
            elif skip_duplicate_spans and (span in spans[argument.sent_index]):
                #z = [ent for ent in entities[argument.sent_index] if tuple(ent[1:]) == s]
                #logging.warn(f"Entities with same span.\n\tid:\t{doc.id}.\n\tSkipping:\t{argument}\n\tDuplicates:\t{z}")
                entity_counter[(argument.type_, 0)] += 1
            else:
                entities[argument.sent_index].append(entity)
                subtypes[argument.sent_index].append(subtype)
                spans[argument.sent_index].add(span)
                entity_counter[(argument.type_, 1)] += 1



    # get indices for entities
    entity_indices = [OrderedDict() for _ in range(sent_count)]
    for i, sent in enumerate(entities):
        for j, entity in enumerate(sent):
            entity_indices[i][entity] = j

    # map events to relations
    relations = [[] for _ in range(sent_count)]
    relation_counter = Counter()
    for event in events:
        for i, argument in enumerate(event.arguments):
            # retain trigger
            if i == 0:
                trigger = argument


            # create relation
            else:

                head = (trigger.type_,  trigger.token_start,  trigger.token_end)
                tail = (argument.type_, argument.token_start, argument.token_end)
                ent_idx = entity_indices[trigger.sent_index]

                if (trigger.sent_index == argument.sent_index) and \
                   (head in ent_idx) and (tail in ent_idx):

                    head_index = ent_idx[head]
                    tail_index = ent_idx[tail]
                    relations[trigger.sent_index].append((head_index, tail_index))
                    relation_counter[(trigger.type_, argument.type_, 1)] += 1


                else:
                    #logging.warn(f"Trigger not an same sentence as arguemnt.\n\tid:\t{doc.id}\n Trigger:\t{trigger}\nArgument:\t{argument}")

                    relation_counter[(trigger.type_, argument.type_, 0)] += 1

    assert len(entities) == len(subtypes)
    assert len(entities) == len(relations)
    assert len(entities) == len(doc.tokens)
    assert len(entities) == len(doc.token_offsets)

    spert_doc = []
    for i, (T, O, E, S, R) in enumerate(zip(doc.tokens, doc.token_offsets, entities, subtypes, relations)):
        sent = {}
        sent[ID] = doc.id

        if include_doc_text and (i == 0):
            sent[DOC_TEXT] = doc.text
        else:
            sent[DOC_TEXT] = None

        sent[SENT_INDEX] = i

        assert len(T) == len(O)
        sent[TOKENS] = T
        sent[OFFSETS] = O

        sent[ENTITIES] = [create_entity(t, s, e) for t, s, e in E]
        sent[SUBTYPES] = [create_entity(t, s, e) for t, s, e in S]
        sent[RELATIONS] = [create_relation(h, t) for h, t in R]


        spert_doc.append(sent)
    return (spert_doc, entity_counter, relation_counter)





def get_entity(entity, subtype, tokens, offsets, indices):

    start = entity[START]
    end = entity[END]

    # get entity tokens
    entity_tokens = tokens[start:end]

    # get entity character span indices
    chars = offsets[start:end]
    char_start = chars[0][0]
    char_end = chars[-1][-1]
    char_count = char_end - char_start

    # get text
    text = [" "]*char_count
    for (s, e), t in zip(chars, entity_tokens):
        text[s-char_start:e-char_start] = t
    text = "".join(text)

    # get token indices
    token_indices = indices[start:end]
    token_start = token_indices[0]
    token_end = token_indices[-1] + 1

    # check text vs tokens
    text_no_ws = "".join(text.split())
    tokens_no_ws = "".join("".join(entity_tokens).split())
    assert len(text_no_ws) > 0
    assert text_no_ws == tokens_no_ws

    # check tokens vs indices
    assert (token_end - token_start) == len(entity_tokens), f'{(token_end - token_start)} vs {len(entity_tokens)}'

    # get entity type
    entity_type = entity[TYPE]

    # get entity subtype
    entity_subtype = None
    if (subtype is not None) and (entity[TYPE] != subtype[TYPE]):
        entity_subtype = subtype[TYPE]

    # create entity object
    return Entity( \
                type_ = entity_type,
                char_start = char_start,
                char_end = char_end,
                text = text,
                subtype = entity_subtype,
                tokens = entity_tokens,
                token_start = token_start,
                token_end = token_end,
                sent_index = None)

def get_entities(spert_entities, spert_subtypes, tokens, offsets, indices):

    # get sub types as dict for fast look up
    subtype_dict = get_subtype_dict(spert_subtypes)

    # process entities
    # iterate over entities in sentence
    entities = []
    for i, spert_entity in enumerate(spert_entities):

        # get applicable subtype
        spert_subtype = None
        k = entity_indices(spert_entity)
        if k in subtype_dict:
            spert_subtype = subtype_dict[k]

        # build entity
        entity = get_entity(spert_entity, spert_subtype, tokens, offsets, indices)

        entities.append(entity)

    return entities


def get_relations(entities, spert_relations):

    relations = []
    for spert_relation in spert_relations:
        type_ = spert_relation[TYPE]
        head = spert_relation[HEAD]
        tail = spert_relation[TAIL]
        relations.append((type_, entities[head], entities[tail]))

    return relations


def get_events(relations):
    """
    Build events from relations
    """
    # iterate over relations
    event_dict = OrderedDict()
    for type_, trigger, argument in relations:

        # create temporary key for aggregate in arguments in events
        k = entity_to_trigger_key(trigger)

        # initialize event
        if k not in event_dict:
            event_dict[k] = Event( \
                                type_ = trigger.type_,
                                arguments = [trigger])

        # include argument
        event_dict[k].arguments.append(argument)

    # convert to list
    events = list(event_dict.values())

    return events


def get_subtype_dict(subtypes):

    # iterate over list of SUBTYPES
    subtype_dict = OrderedDict()
    for subtype in subtypes:
        k = entity_indices(subtype)
        subtype_dict[k] = subtype
    return subtype_dict

def entity_indices(entity):
    return (entity[START], entity[END])


def entity_to_trigger_key(entity):
    return (entity.type_, entity.token_start, entity.token_end)


def spert2doc(spert_doc):
    """
    Create Document object from spurt output
    """

    # iterate over sentences in doc
    token_count = 0
    doc_tokens = []
    doc_offsets = []
    doc_entities = []
    doc_events = []
    for sent in spert_doc:

        # deconstruct sentence encoding
        id = sent[ID]
        tokens = sent[TOKENS]
        offsets = sent[OFFSETS]
        spert_entities = sent[ENTITIES]
        spert_subtypes = sent[SUBTYPES]
        spert_relations = sent[RELATIONS]

        # get sentence length
        sent_length = len(tokens)

        # get token indices at document level
        indices = [i + token_count for i, _ in enumerate(tokens)]

        # increment token counter
        token_count += sent_length

        # get entities
        entities = get_entities(spert_entities, spert_subtypes, tokens, offsets, indices)
        doc_entities.extend(entities)

        relations = get_relations(entities, spert_relations)

        events = get_events(relations)
        doc_events.extend(events)

        doc_tokens.append(tokens)
        doc_offsets.append(offsets)

    doc = DocumentPredict( \
                    id = id,
                    entities = doc_entities,
                    events = doc_events,
                    tokens = doc_tokens,
                    offsets = doc_offsets)
    return doc


def spert2corpus(input_file):
    """
    Create Corpus object from spert ouput
    """

    # load spert output
    spert_corpus = json.load(open(input_file, "r"))

    # aggregate sentences by document
    # iterate over sentences in corpus
    by_doc = OrderedDict()
    for sent in spert_corpus:

        # get
        id = sent[ID]

        # initialize current document
        if id not in by_doc:
            by_doc[id] = []

        by_doc[id].append(sent)

    # process documents
    # iterate over documents in corpus
    corpus = CorpusPredict()
    for id, spert_doc in by_doc.items():

        doc = spert2doc(spert_doc)
        corpus.add_doc(doc)

    return corpus

def get_next_id(d, k):
    d[k] += 1
    return f'{k}{d[k]}'

def get_next_tb(d, k='T'):
    return get_next_id(d, k)

def get_next_event(d, k='E'):
    return get_next_id(d, k)

def get_next_attr(d, k='A'):
    return get_next_id(d, k)

def get_span2tb(spert_entities, ids):



    span2tb = OrderedDict()
    entity2tb = OrderedDict()
    for i, entity in enumerate(spert_entities):

        span = (entity[START], entity[END])

        tb = get_next_tb(ids)

        assert span not in span2tb
        span2tb[span] = tb

        assert i not in entity2tb
        entity2tb[i] = tb

    return (span2tb, entity2tb)


def get_entity_dict(spert_entities, span2tb, require_span_match=True, ignore=None):


    entity_dict = {}
    for entity in spert_entities:
        type_ = entity[TYPE]
        span = (entity[START], entity[END])

        if (ignore is not None) and (type_ in ignore):
            pass
        elif span in span2tb:
            tb = span2tb[span]
            entity_dict[tb] = entity
        elif require_span_match:
            raise ValueError(f"tb {span} not in span2tb {span2tb.keys()}")
        else:
            pass

    return entity_dict


def get_relation_dict(spert_relations, entity2tb, require_span_match=True):

    relation_dict = {}
    for relation in spert_relations:

        head_idx = relation[HEAD]
        tail_idx = relation[TAIL]

        head_tb = entity2tb[head_idx]
        tail_tb = entity2tb[tail_idx]

        relation_dict[(head_tb, tail_tb)] = relation
    return relation_dict


def get_tb_dict(entity_dict, offsets, text):

    # iterate over entities in entity_dict
    tb_dict = {}
    for tb_id, entity in entity_dict.items():
        type_ = entity[TYPE]

        # get token indices
        token_start = entity[START]
        token_end = entity[END]

        # get character indices from token indices
        char_offsets = offsets[token_start:token_end]
        char_start = char_offsets[0][0]
        char_end = char_offsets[-1][1]

        text_span = text[char_start:char_end]

        tb = Textbound(
            id = tb_id,
            type_ = type_,
            start = char_start,
            end = char_end,
            text = text_span,
        )
        tb_dict[tb_id] = tb
    return tb_dict

def get_event_dict(relation_dict, entity_dict, ids):


    # collect arguments by trigger
    args_by_trigger = {}
    for (trig_id, arg_id) in relation_dict:
        if trig_id not in args_by_trigger:
            args_by_trigger[trig_id] = []
        args_by_trigger[trig_id].append(arg_id)

    # iterate over trigger text bounds
    event_dict = {}
    for trig_id, arg_ids in args_by_trigger.items():

        # build the arguments with trigger
        arguments = OrderedDict()

        # get trigger information
        trigger = entity_dict[trig_id]
        trigger_type = trigger[TYPE]
        arguments[trigger_type] = trig_id

        # iterate over argument ids
        for arg_id in arg_ids:

            # get argument information
            argument = entity_dict[arg_id]
            argument_type = argument[TYPE]

            # add arguments to event, checking for argument type uniqueness
            argument_type = get_unique_arg(argument_type, arguments)
            assert argument_type not in arguments
            arguments[argument_type] = arg_id

        # create a event
        event = Event( \
                    id = get_next_event(ids),
                    type_ = trigger_type,
                    arguments = arguments)

        # bundle events
        assert event.id not in event_dict
        event_dict[event.id] = event

    return event_dict


def merge_subtype_with_entity(subtype_dict, entity_dict, relation_dict, argument_pairs, ids):

    for tb, subtype in subtype_dict.items():

        # get associated entity and entity type
        entity = entity_dict[tb]
        entity_type = entity[TYPE]

        # get new entity type from mapping
        assert entity_type in argument_pairs, \
            f'entity_type "{entity_type}" not in argument_pairs "{argument_pairs.keys()}"'
        new_entity_type = argument_pairs[entity_type]

        # create new entity (similar to SpERT entity, but VALUE added)
        new_entity = { \
                        TYPE: new_entity_type,
                        START: subtype[START],
                        END:   subtype[END],
                        VALUE: subtype[TYPE]}

        tb_new = get_next_tb(ids)

        assert tb_new not in entity_dict
        entity_dict[tb_new] = new_entity

        relation_dict[(tb, tb_new)] = {TYPE: RELATION_DEFAULT}

    return (subtype_dict, entity_dict, relation_dict)

def get_attr_dict(entity_dict, ids):

    print(entity_dict)
    attr_dict = {}
    for tb_id, entity in entity_dict.items():
        if VALUE in entity:
            attr = Attribute( \
                    id = get_next_attr(ids),
                    type_ = entity[TYPE],
                    textbound = tb_id,
                    value = entity[VALUE])
            print(attr)
            attr_dict[tb_id] = attr

    return attr_dict

def spert_sent2brat_dicts(spert_sent, argument_pairs, text, ids):

    id = spert_sent[ID]
    tokens = spert_sent[TOKENS]
    offsets = spert_sent[OFFSETS]
    entities = spert_sent[ENTITIES]
    subtypes = spert_sent[SUBTYPES]
    relations = spert_sent[RELATIONS]


    span2tb, entity2tb = get_span2tb(entities, ids)

    #print(span2tb)
    entity_dict = get_entity_dict(entities, span2tb, require_span_match=True, ignore=None)
    subtype_dict = get_entity_dict(subtypes, span2tb, require_span_match=False, ignore=[SUBTYPE_DEFAULT])
    relation_dict = get_relation_dict(relations, entity2tb, require_span_match=True)



    # iterate over spans labele with subtype

    subtype_dict, entity_dict, relation_dict = merge_subtype_with_entity( \
                    subtype_dict, entity_dict, relation_dict, argument_pairs, ids)


    tb_dict = get_tb_dict(entity_dict, offsets=offsets, text=text)
    event_dict = get_event_dict(relation_dict, entity_dict, ids)
    attr_dict = get_attr_dict(entity_dict, ids)


    # print(tb_dict)
    # print(event_dict)





def spert_doc2brat_dicts(spert_doc, argument_pairs):

    #print(spert_doc)

    # make sure not empty
    assert len(spert_doc) > 0

    # get text
    text = spert_doc[0][DOC_TEXT]
    assert text is not None
    assert len(text) > 0


    event_dict = {}
    relation_dict = None
    tb_dict = {}
    attr_dict = {}
    #print(argument_pairs)

    ids = {'T':0, 'E':0, 'A':0}


    for i, spert_sent in enumerate(spert_doc):
        assert spert_sent[SENT_INDEX] == i
        print('-'*80)
        print(i)
        spert_sent2brat_dicts(spert_sent, argument_pairs, text, ids)




    #
    # SUBTYPES = "subtypes"
    # TYPE = "type"
    # START = "start"
    # END = "end"
    # HEAD = "head"
    # TAIL = "tail"


    return (event_dict, relation_dict, tb_dict, attr_dict)

def spert2doc_dict(input_file):


    tokenizer = get_tokenizer()

    # load spert output
    spert_corpus = json.load(open(input_file, "r"))

    # aggregate sentences by document
    # iterate over sentences in corpus
    by_doc = OrderedDict()
    for sent in spert_corpus:

        # get
        id = sent[ID]

        # initialize current document
        if id not in by_doc:
            assert sent[SENT_INDEX] == 0
            assert sent[DOC_TEXT] is not None
            assert len(sent[DOC_TEXT]) > 0

            by_doc[id] = []

        by_doc[id].append(sent)


    return by_doc


def merge_spert_encodings(original, predict, fields=FIELDS):
    """
    Merge original and predicted spert IO
    """

    assert len(original) == len(predict)

    # iterate over our sentences
    merged = []
    for orig, preds in zip(original, predict):

        assert orig[TOKENS] == preds[TOKENS]

        # add unnecessary original fields to prediction
        d = {}
        d.update(preds)
        for field in fields:
            if field in orig:
                d[field] = orig[field]

        merged.append(d)

    return merged


def merge_spert_files(original_file, predict_file, merged_file, fields=FIELDS):
    """
    Load and merge original and predicted spert IO
    """

    # load files
    original = json.load(open(original_file, 'r'))
    predict = json.load(open(predict_file, 'r'))

    # merge data structures
    merged = merge_spert_encodings(original, predict, fields=fields)

    # save merged data structure
    json.dump(merged, open(merged_file, "w"))

    return merged





#def swap_subtype2type(file_in, file_out, subtype_default=SUBTYPE_DEFAULT):
def swap_type2subtype(file_in, file_out, subtype_val=None):

    doc = json.load(open(file_in, 'r'))

    for i, sent in enumerate(doc):
        orig_entities = copy.deepcopy(sent[ENTITIES])
        orig_subtypes = copy.deepcopy(sent[SUBTYPES])

        sent[ENTITIES] = orig_subtypes

        if subtype_val is not None:
            for x in orig_entities:
                x[TYPE] = subtype_val
        sent[SUBTYPES] = orig_entities

        #for s in sent[SUBTYPES]:
        #    s[TYPE] = subtype_default

    json.dump(doc, open(file_out, 'w'))

# def swap_type2subtype(file_in, file_out, map_):
#
#     doc = json.load(open(file_in, 'r'))
#
#     for i, sent in enumerate(doc):
#         sent[SUBTYPES] = copy.deepcopy(sent[ENTITIES])
#
#         for d in sent[ENTITIES]:
#             d[TYPE] = map_[d[TYPE]]
#
#     json.dump(doc, open(file_out, 'w'))

def map_type2subtype(file_in, file_out, map_):

    doc = json.load(open(file_in, 'r'))

    for i, sent in enumerate(doc):
        sent[SUBTYPES] = copy.deepcopy(sent[ENTITIES])

        for d in sent[ENTITIES]:
            d[TYPE] = map_[d[TYPE]]

    json.dump(doc, open(file_out, 'w'))



def plot_loss(input_file, destination_file=None, iteration_column='global_iteration', loss_column='loss_avg'):


    df = pd.read_csv(input_file)

    x = df[iteration_column]
    y = df[loss_column]

    # create figure
    fig, ax = plt.subplots(1)

    ax.plot(x, y)
    ax.set_ylabel('loss')
    ax.set_xlabel('iteration')

    if destination_file is None:
        destination_file = Path(input_file).with_suffix('.png')


    fig.savefig(destination_file)

    plt.close(fig=fig)
    plt.close('all')

    return True
