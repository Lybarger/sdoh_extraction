from collections import OrderedDict, Counter
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import logging
import config.constants as constants
import copy

from corpus.utils import remove_white_space_at_ends

from config.constants import TRIGGER


class Entity(object):
    '''
    '''
    def __init__(self, type_, char_start, char_end, text, \
        subtype=None, tokens=None, token_start=None, token_end=None, sent_index=None):

        self.type_ = type_
        self.char_start = char_start
        self.char_end = char_end
        self.text = text
        self.subtype = subtype
        self.tokens = tokens
        self.token_start = token_start
        self.token_end = token_end
        self.sent_index = sent_index

    def indices(self):
        return (self.char_start, self.char_end)

    def __str__(self):
        x = ['{}={}'.format(k, v) for k, v in self.__dict__.items()]
        x = ', '.join(x)
        x = 'Entity({})'.format(x)
        return x

    def as_tuple(self):
        return tuple([v for k, v in self.__dict__.items()])


    def strip(self):

        self.text, self.char_start, self.char_end = \
                    remove_white_space_at_ends(self.text, self.char_start, self.char_end)


    def get_key(self):
        return  self.value()

    def value(self):
        return  (self.char_start, self.char_end, self.type_, self.subtype)

    def __eq__(self, other):
        return self.value() == other.value()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash('-'.join([str(x) for x in self.value()]))

    def __lt__(self, other):
        return self.value() < other.value()

class Relation(object):
    '''
    '''
    def __init__(self, entity_a, entity_b, role):
        self.entity_a = entity_a
        self.entity_b = entity_b
        self.role = role

    def __str__(self):
        x = ['{}={}'.format(k, v) for k, v in self.__dict__.items()]
        x = ', '.join(x)
        x = 'Relation({})'.format(x)
        return x

    def strip(self):

        self.entity_a.strip()
        self.entity_b.strip()


class Event(object):
    '''
    '''
    def __init__(self, type_, arguments):
        self.type_ = type_

        assert isinstance(arguments, list)
        self.arguments = arguments

    def __str__(self):
        x = f'Event(type_={self.type_}, arguments=['
        for arg in self.arguments:
            x += '\n\t' + str(arg)
        x += '])'
        return x




def get_indices_by_sent(start, end, offsets, tokens):
    """
    Get sentence index for textbounds
    """

    # iterate over sentences
    sent_start = None
    sent_end = None
    token_start = None
    token_end = None

    for i, sent in enumerate(offsets):

        for j, (char_start, char_end) in enumerate(sent):

            if (start >= char_start) and (start <  char_end):
                sent_start = i
                token_start = j
            if (end   >  char_start) and (end   <= char_end):
                sent_end = i
                token_end = j + 1

    assert sent_start is not None
    assert sent_end is not None
    assert token_start is not None
    assert token_end is not None

    if (sent_start != sent_end):
        logging.warn(f"Entity spans multiple sentences, truncating")
        token_end = len(offsets[sent_start])

    toks = tokens[sent_start][token_start:token_end]

    return (sent_start, token_start, token_end, toks)

def get_indices_by_doc(start, end, offsets, tokens):
    """
    Get sentence index for textbounds
    """

    # iterate over sentences
    token_start = None
    token_end = None

    # flatten offsets
    offsets = [idx for sent in offsets for idx in sent]
    tokens =  [tok for sent in tokens  for tok in sent]

    for j, (char_start, char_end) in enumerate(offsets):

        if (start >= char_start) and (start <  char_end):
            token_start = j
        if (end   >  char_start) and (end   <= char_end):
            token_end = j + 1

    assert token_start is not None
    assert token_end is not None

    toks = tokens[token_start:token_end]

    return (None, token_start, token_end, toks)



def get_indices(start, end, offsets, tokens, by_sent=False):
    """
    Get sentence index for textbounds
    """

    if by_sent:
        sent_index, token_start, token_end, toks = get_indices_by_sent(start, end, offsets, tokens)
    else:
        sent_index, token_start, token_end, toks = get_indices_by_doc(start, end, offsets, tokens)

    return (sent_index, token_start, token_end, toks)

def tb2entities(tb_dict, attr_dict, \
                        as_dict = False,
                        tokens = None,
                        token_offsets = None,
                        by_sent = False):
    """
    convert textbound add attribute dictionaries to entities
    """

    # iterate over textbounds
    entities = OrderedDict()
    for tb_id, tb in tb_dict.items():

        subtype = None

        if tb_id in attr_dict:
            attr = attr_dict[tb_id]
            subtype = attr.value

            if tb.type_ not in attr.type_:
                logging.warn(f"possible attribute matching error: {tb.type_} vs {attr.type_}")

        if (tokens is None) or (token_offsets is None):
            sent_index = None
            token_indices_ = None
            tokens_ = None
        else:
            sent_index, token_start, token_end, tokens_ = get_indices( \
                                start = tb.start,
                                end = tb.end,
                                offsets = token_offsets,
                                tokens = tokens,
                                by_sent = by_sent)

        # create entity
        entity = Entity( \
            type_ = tb.type_,
            char_start = tb.start,
            char_end = tb.end,
            text = tb.text,
            subtype = subtype,
            tokens = tokens_,
            token_start = token_start,
            token_end = token_end,
            sent_index = sent_index)


        assert tb_id not in entities
        entities[tb_id] = entity

    if as_dict:
        return entities
    else:
        return [entity for _, entity in entities.items()]

def tb2relations(relation_dict, tb_dict, attr_dict, \
                        as_dict = False,
                        tokens = None,
                        token_offsets = None,
                        by_sent = False):
    """
    convert textbound and relations to relation object
    """

    # get entities from textbounds
    entities = tb2entities(tb_dict, attr_dict, \
                            as_dict = True,
                            tokens = tokens,
                            token_offsets = token_offsets,
                            by_sent = by_sent)

    # iterate over a relation dictionary
    relations = OrderedDict()


    for id, relation_brat in relation_dict.items():

        tb_1 = relation_brat.arg1
        tb_2 = relation_brat.arg2
        role = relation_brat.role


        if (tb_1[0] == 'E') or (tb_2[0] == 'E'):
            logging.warn(f"tb2relations - Relation defined between events. Cannot accommodate: {id} - {relation_brat}")
        else:
            assert tb_1 in entities, f'reltation tb {tb_1} not in entities {entities.keys()}'
            assert tb_2 in entities, f'reltation tb {tb_2} not in entities {entities.keys()}'

            relation = Relation( \
                    entity_a = copy.deepcopy(entities[tb_1]),
                    entity_b = copy.deepcopy(entities[tb_2]),
                    role = role)

            assert id not in relations
            relations[id] = relation

    if as_dict:
        return relations
    else:
        return [relation for _, relation in relations.items()]


def brat2events(event_dict, tb_dict, attr_dict, \
                        as_dict = False,
                        tokens = None,
                        token_offsets = None,
                        by_sent = False):
    """
    convert textbound and relations to relation object
    """




    # get entities from textbounds
    entities = tb2entities(tb_dict, attr_dict, \
                            as_dict = True,
                            tokens = tokens,
                            token_offsets = token_offsets,
                            by_sent = by_sent)

    # iterate over a relation dictionary
    events = OrderedDict()
    for id, event_brat in event_dict.items():

        # iterate over arguments
        arguments = []
        for i, (argument_type, tb_id) in enumerate(event_brat.arguments.items()):

            entity = copy.deepcopy(entities[tb_id])

            # assume first entity is the trigger
            #if i == 0:
            #    entity.subtype = entity.type_
            #    entity.type_ = TRIGGER

            arguments.append(entity)

        event = Event( \
                type_ = event_brat.type_,
                arguments = arguments)

        assert id not in events
        events[id] = event

    if as_dict:
        return events
    else:
        return [event for _, event in events.items()]




def event2relations(event, role=1):

    trigger = event.arguments[0]
    assert trigger.type_ == TRIGGER

    relations = []
    for argument in event.arguments[1:]:
        relation = Relation(entity_a=trigger, entity_b=argument, role=role)
        relations.append(relation)

    return relations

def events2relations(events):

    relations = []
    for event in events:
        relations.extend(event2relations(event))

    return relations


def relations2events(relations, out_type='list'):

    # iterate over relations
    event_dict = OrderedDict()
    for relation in relations:

        # assume first entity is trigger and second entity is argument
        trigger = relation.entity_a
        argument = relation.entity_b

        # trigger representation that suitable for dictionary key
        trigger_key = trigger.get_key()

        # If new trigger, create new event
        if trigger_key not in event_dict:
            event_dict[trigger_key] = Event( \
                            type_ = trigger.subtype,
                            arguments = [trigger]
                            )

        # add argument to existing event
        event_dict[trigger_key].arguments.append(argument)

    if out_type == 'dict':
        events = event_dict
    elif out_type == 'list':
        events = list(event_dict.values())
    else:
        raise ValueError("invalid out_type")


    return events
