
import argparse
import json
from collections import OrderedDict

import spacy
from tqdm import tqdm
import os
from pathlib import Path
import re
import logging
import re
import string


#from corpus.brat import get_annotations, import_brat_dir


SPERT_ID = "id"
SPERT_TOKENS = "tokens"
SPERT_ENTITIES = "entities"
SPERT_RELATIONS = "relations"
SPERT_SUBTYPES = "subtypes"
SPERT_TYPE = "type"
SPERT_START = "start"
SPERT_END = "end"
SPERT_HEAD = "head"
SPERT_TAIL = "tail"


RELATION_DEFAULT = 'relation'
CHAR_COUNT = 12


def rm_ws(spacy_tokens):
    return [token for token in spacy_tokens if token.text.strip()]

def tokenize_document(text, tokenizer):

    doc = tokenizer(text)

    #sent_bounds = []
    tokens = []
    offsets = []
    for sent in rm_ws(doc.sents):
        #sent_bounds.append((sent.start_char, sent.end_char))
        sent = rm_ws(sent)

        tok = [t.text for t in sent]
        os = [(t.idx, t.idx + len(t.text)) for t in sent]

        tokens.append(tok)
        offsets.append(os)

    # Check
    for tok, off in zip(tokens, offsets):
        for t, o in zip(tok, off):
            assert t == text[o[0]:o[1]]

    #return (sent_bounds, tokens, offsets)
    return (tokens, offsets)


def start_match(x_start, y_start, y_end):
    '''
    Determine if x is in range of y
    x_start:

    '''
    return (x_start >= y_start) and (x_start <  y_end)

def end_match(x_end, y_start, y_end):
    '''
    Determine if x is in range of y
    x_start:

    '''

    return (x_end   >  y_start) and (x_end   <= y_end)


def get_tb_indices(tb_dict, offsets):
    """
    Get sentence index for textbounds
    """

    map = {}

    # iterate over text bounds
    for tb_id, tb in tb_dict.items():
        assert tb.id == tb_id

        # iterate over sentences
        sent_index = None
        token_start_index = None
        token_end_index = None
        for i, sent_offsets in enumerate(offsets):

            sent_start = sent_offsets[0][0]
            sent_end = sent_offsets[-1][-1]

            sent_start_match = start_match(tb.start, sent_start, sent_end)
            sent_end_match =   end_match(tb.end,     sent_start, sent_end)

            # text bound in sentence
            if sent_start_match:
                sent_index = i

                if not sent_end_match:
                    logging.warn(f"Textbound end not in same sentences start: {tb}")

                # iterate over tokens
                for j, (token_start, token_end) in enumerate(sent_offsets):
                    if start_match(tb.start, token_start, token_end):
                        token_start_index = j
                    if end_match(tb.end, token_start, token_end):
                        token_end_index = j + 1
                break

        assert sent_index is not None
        assert token_start_index is not None
        if token_end_index is None:
            logging.warn(f"Token end index is None")
            token_end_index = len(offsets[sent_index])

        map[tb_id] = (sent_index, token_start_index, token_end_index)

    return map

def convert_doc(text, ann, id, tokenizer, \
                    allowable_tb = None,
                    relation_default = RELATION_DEFAULT):


    tokens, offsets = tokenize_document(text, tokenizer)


    # Extract events, text bounds, and attributes from annotation string
    event_dict, relation_dict, tb_dict, attr_dict = get_annotations(ann)

    indices = get_tb_indices(tb_dict, offsets)

    sent_count = len(tokens)

    entities = [OrderedDict() for _ in range(sent_count)]
    subtypes = [OrderedDict() for _ in range(sent_count)]
    relations = [[] for _ in range(sent_count)]


    for event_id, event in event_dict.items():
        #print()
        #print(event_id, event)

        head_tb_id = None
        head_sent = None
        head_index = None

        for i, (tb_type, tb_id) in enumerate(event.arguments.items()):
            #print()

            sent_index, token_start, token_end = indices[tb_id]
            #print("sentence index", sent_index, token_start, token_end)
            #print(tb_type, tb_id)
            tb = tb_dict[tb_id]
            #print(tb)


            if tb_id in attr_dict:
                attr_type = attr_dict[tb_id].type_
                attr_value = attr_dict[tb_id].value
            else:
                attr_type = tb.type_
                attr_value = tb.type_

            if tb.type_ not in attr_type:
                logging.warn(f"Attribute type not in textbound type: {tb.type_} not in {attr_type}")


            #print(attr)
            if (allowable_tb is None) or (tb.type_ in allowable_tb):

                if tb_id not in entities[sent_index]:
                    d = {SPERT_TYPE: tb.type_, SPERT_START: token_start, SPERT_END: token_end}
                    entities[sent_index][tb_id] = d

                    d = {SPERT_TYPE: attr_value, SPERT_START: token_start, SPERT_END: token_end}
                    subtypes[sent_index][tb_id] = d

                entity_index = list(entities[sent_index].keys()).index(tb_id)


                if i == 0:
                    head_tb_id = tb_id
                    head_sent = sent_index
                    head_index = entity_index
                elif head_sent == sent_index:

                    assert head_tb_id is not None
                    assert head_sent is not None
                    assert head_index is not None

                    d = {SPERT_TYPE: relation_default, SPERT_HEAD: head_index, SPERT_TAIL: entity_index}
                    relations[sent_index].append(d)

                else:
                    logging.warn(f"Head index not an same sentence as tail. Skipping relation.")


    #print(entities)
    entities = [list(sent.values()) for sent in entities]
    subtypes = [list(sent.values()) for sent in subtypes]
    #print(entities)

    #print(relations)

    assert len(tokens) == sent_count
    assert len(entities) == sent_count
    assert len(relations) == sent_count

    out = []
    for i in range(sent_count):
        d = {}
        d[SPERT_ID] = f'{id}[{i}]'
        d[SPERT_TOKENS] = tokens[i]
        d[SPERT_ENTITIES] = entities[i]
        d[SPERT_SUBTYPES] = subtypes[i]
        d[SPERT_RELATIONS] = relations[i]
        out.append(d)

    return out

def format_str(x, n=CHAR_COUNT):

    y = x[0:n].ljust(n)
    return y





def format_doc(doc):

    out = []
    for sent in doc:
        out.append('')
        out.append(sent[SPERT_ID])

        tokens = sent[SPERT_TOKENS]
        n = len(tokens)
        tokens = ' '.join([format_str(y) for y in ['TOKENS:'] + tokens])
        out.append(tokens)

        entities = ['']*n
        for entity in sent[SPERT_ENTITIES]:
            start = entity[SPERT_START]
            end = entity[SPERT_END]
            type = entity[SPERT_TYPE]
            for i in range(start, end):
                if i == start:
                    entities[i] = type
                elif i == end -1:
                    entities[i] = '-'*(CHAR_COUNT-1) + '>'
                else:
                    entities[i] = '-'*CHAR_COUNT
        entities = ' '.join([format_str(y) for y in ['ENTITIES:'] + entities])
        out.append(entities)

        relations = ['']*n
        for relation in sent[SPERT_RELATIONS]:

            type = relation[SPERT_TYPE]
            head = relation[SPERT_HEAD]
            tail = relation[SPERT_TAIL]



            if tail > head:

                start = sent[SPERT_ENTITIES][head][SPERT_START]
                end = sent[SPERT_ENTITIES][tail][SPERT_END]

                for i in range(start, end):
                    if i == start:
                        relations[i] = entity[SPERT_TYPE]
                    elif i == end -1:
                        relations[i] = '-'*(CHAR_COUNT-1) + '>'
                    else:
                        relations[i] = '-'*CHAR_COUNT
            else:

                start = sent[SPERT_ENTITIES][head][SPERT_END] - 1
                end = sent[SPERT_ENTITIES][tail][SPERT_START]

                for i in range(start, end-1, -1):
                    if i == start:
                        relations[i] = entity[SPERT_TYPE]
                    elif i == end:
                        relations[i] = '<' + '-'*(CHAR_COUNT-1)
                    else:
                        relations[i] = '-'*CHAR_COUNT

        relations = ' '.join([format_str(y) for y in ['RELATIONS:'] + relations])
        out.append(relations)

    out = '\n'.join(out)

    return out

def get_allowable_types(path):

    if path is None:
        return None
    else:
        types = json.load(open(path, 'r'))
        #d = {}
        #d['entities'] = list(types['entities'].keys())
        #d['relations'] = list(types['relations'].keys())

        allowable_tb = list(types['entities'].keys())
        return allowable_tb


def convert_brat(source_path, dest_path, spacy_model='en_core_web_sm', types_path=None, sample_count=None):

    tokenizer = spacy.load(spacy_model)


    allowable_tb = get_allowable_types(types_path)


    annotations = import_brat_dir(path=source_path, sample_count=sample_count)


    logging.info(f"")
    logging.info(f"Converting documents: {source_path}")
    pbar = tqdm(total=len(annotations))

    # Loop on annotated files
    converted_docs = []
    formatted_docs = []
    for id, text, ann in annotations:

        doc = convert_doc(text, ann, id, tokenizer, allowable_tb=allowable_tb)

        converted_docs.extend(doc)
        formatted_docs.append(format_doc(doc))

        pbar.update(1)
    pbar.close()

    json.dump(converted_docs, open(dest_path, 'w'))

    formatted_doc_path = Path(dest_path).with_suffix('.txt')
    with open(formatted_doc_path, 'w') as f:
        formatted_docs = '\n'.join(formatted_docs)
        f.write(formatted_docs)



    return converted_docs


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--source_path', type=str, help="Path to dataset")
    arg_parser.add_argument('--dest_path', type=str, help="Destination file path (JSON format)")
    arg_parser.add_argument('--spacy_model', type=str, default='en_core_web_sm', help="SpaCy model")
    arg_parser.add_argument('--types_path', type=str, default=None, help="Types file path (JSON format)")
    arg_parser.add_argument('--sample_count', type=int, default=-1, help="Number of files to load")

    args = arg_parser.parse_args()
    convert(args.source_path, args.dest_path, args.spacy_model, args.types_path, args.sample_count)






# TEXT_FILE_EXT = 'txt'
# ANN_FILE_EXT = 'ann'
# ENCODING = 'utf-8'

# COMMENT_RE = re.compile(r'^#')
# TEXTBOUND_RE = re.compile(r'^T\d+')
# EVENT_RE = re.compile(r'^E\d+\t')
# ATTRIBUTE_RE = re.compile(r'^A\d+\t')
# RELATION_RE = re.compile(r'^R\d+\t')



# def get_filename(path):
#     root, ext = os.path.splitext(path)
#     return root
#
# def filename_check(fn1, fn2):
#     '''
#     Confirm filenames, regardless of directory or extension, match
#     '''
#     fn1 = get_filename(fn1)
#     fn2 = get_filename(fn2)
#
#     return fn1==fn2

# def get_files(path, ext='.', relative=False):
#     files = list(Path(path).glob('**/*.{}'.format(ext)))
#
#     if relative:
#         files = [os.path.relpath(f, path) for f in files]
#
#     return files


# def get_brat_files(path):
#     '''
#     Find text and annotation files
#     '''
#     # Text and annotation files
#     text_files = get_files(path, TEXT_FILE_EXT, relative=False)
#     ann_files = get_files(path, ANN_FILE_EXT, relative=False)
#
#     # Check number of text and annotation files
#     msg = 'Number of text and annotation files do not match'
#     assert len(text_files) == len(ann_files), msg
#
#     # Sort files
#     text_files.sort()
#     ann_files.sort()
#
#     # Check the text and annotation filenames
#     mismatches = [str((t, a)) for t, a in zip(text_files, ann_files) \
#                                            if not filename_check(t, a)]
#     fn_check = len(mismatches) == 0
#     assert fn_check, '''txt and ann filenames do not match:\n{}'''. \
#                         format("\n".join(mismatches))
#
#     return (text_files, ann_files)
#




# class Attribute(object):
#     '''
#     Container for attribute
#
#     annotation file examples:
#         A1      Value T2 current
#         A2      Value T8 current
#         A3      Value T9 none
#         A4      Value T13 current
#         A5      Value T17 current
#     '''
#     def __init__(self, id, type, textbound, value):
#         self.id = id
#         self.type_ = type
#         self.textbound = textbound
#         self.value = value
#
#     def __str__(self):
#         return str(self.__dict__)
#
#     def __eq__(self, other):
#         return (self.type_ == other.attr) and \
#                (self.textbound == other.textbound) and \
#                (self.value == other.value)
#     def brat_str(self):
#         return attr_str(attr_id=self.id, arg_type=self.type_, \
#                             tb_id=self.textbound, value=self.value)
#
#     def id_numerical(self):
#         assert self.id[0] == 'A'
#         id = int(self.id[1:])
#         return id


# class Textbound(object):
#     '''
#     Container for textbound
#
#     Annotation file examples:
#         T2	Tobacco 30 36	smoker
#         T4	Status 38 46	Does not
#         T5	Alcohol 47 62	consume alcohol
#         T6	Status 64 74	No history
#     '''
#     def __init__(self, id, type_, start, end, text):
#         self.id = id
#         self.type_ = type_
#         self.start = start
#         self.end = end
#         self.text = text
#
#     def __str__(self):
#         return str(self.__dict__)
#
#     def token_indices(self, char_indices):
#         i_sent, (out_start, out_stop) = find_span(char_indices, self.start, self.end)
#         return (i_sent, (out_start, out_stop))
#
#     def brat_str(self):
#         return textbound_str(id=self.id, type_=self.type_, start=self.start, \
#                                                 end=self.end, text=self.text)
#
#     def id_numerical(self):
#         assert self.id[0] == 'T'
#         id = int(self.id[1:])
#         return id

# class Event(object):
#     '''
#     Container for event
#
#     Annotation file examples:
#         E3      Family:T7 Amount:T8 Type:T9
#         E4      Tobacco:T11 State:T10
#
#         id     event:head (entities)
#     '''
#
#     def __init__(self, id, type_, arguments):
#         self.id = id
#         self.type_ = type_
#         self.arguments = arguments
#
#     def __str__(self):
#         return str(self.__dict__)
#
#     def brat_str(self):
#         return event_str(id=self.id, event_type=self.type_, \
#                             textbounds=self.arguments)


# class Relation(object):
#     '''
#     Container for event
#
#     Annotation file examples:
#     R1  attr Arg1:T2 Arg2:T1
#     R2  attr Arg1:T5 Arg2:T6
#     R3  attr Arg1:T7 Arg2:T1
#
#     '''
#
#     def __init__(self, id, role, arg1, arg2):
#         self.id = id
#         self.role = role
#         self.arg1 = arg1
#         self.arg2 = arg2
#
#     def __str__(self):
#         return str(self.__dict__)
#
#     def brat_str(self):
#         return relation_str(id=self.id, role=self.role, \
#                             arg1=self.arg1, arg2=self.arg2)

#
# def get_annotations(ann):
#     '''
#     Load annotations, including taxbounds, attributes, and events
#
#     ann is a string
#     '''
#
#     # Parse string into nonblank lines
#     lines = [l for l in ann.split('\n') if len(l) > 0]
#
#
#     # Confirm all lines consumed
#     remaining = [l for l in lines if not \
#             ( \
#                 COMMENT_RE.search(l) or \
#                 TEXTBOUND_RE.search(l) or \
#                 EVENT_RE.search(l) or \
#                 RELATION_RE.search(l) or \
#                 ATTRIBUTE_RE.search(l)
#             )
#         ]
#     msg = 'Could not match all annotation lines: {}'.format(remaining)
#     assert len(remaining)==0, msg
#
#     # Get events
#     events = parse_events(lines)
#
#     # Get relations
#     relations = parse_relations(lines)
#
#     # Get text bounds
#     textbounds = parse_textbounds(lines)
#
#     # Get attributes
#     attributes = parse_attributes(lines)
#
#     return (events, relations, textbounds, attributes)
#
# def parse_textbounds(lines):
#     """
#     Parse textbound annotations in input, returning a list of
#     Textbound.
#
#     ex.
#         T1	Status 21 29	does not
#         T1	Status 27 30	non
#         T8	Drug 91 99	drug use
#
#     """
#
#     textbounds = {}
#     for l in lines:
#         if TEXTBOUND_RE.search(l):
#
#             # Split line
#             id, type_start_end, text = l.split('\t')
#
#             # Check to see if text bound spans multiple sentences
#             mult_sent = len(type_start_end.split(';')) > 1
#
#             # Multiple sentence span, only use portion from first sentence
#             if mult_sent:
#
#                 # type_start_end = 'Drug 99 111;112 123'
#
#                 # type_start_end = ['Drug', '99', '111;112', '123']
#                 type_start_end = type_start_end.split()
#
#                 # type = 'Drug'
#                 # start_end = ['99', '111;112', '123']
#                 type_ = type_start_end[0]
#                 start_end = type_start_end[1:]
#
#                 # start_end = '99 111;112 123'
#                 start_end = ' '.join(start_end)
#
#                 # start_ends = ['99 111', '112 123']
#                 start_ends = start_end.split(';')
#
#                 # start_ends = [('99', '111'), ('112', '123')]
#                 start_ends = [tuple(start_end.split()) for start_end in start_ends]
#
#                 # start_ends = [(99, 111), (112, 123)]
#                 start_ends = [(int(start), int(end)) for (start, end) in start_ends]
#
#                 start = start_ends[0][0]
#
#                 # ends = [111, 123]
#                 ends = [end for (start, end) in start_ends]
#
#                 text = list(text)
#                 for end in ends[:-1]:
#                     n = end - start
#                     assert text[n].isspace()
#                     text[n] = '\n'
#                 text = ''.join(text)
#
#                 start = start_ends[0][0]
#                 end = start_ends[-1][-1]
#
#             else:
#                 # Split type and offsets
#                 type_, start, end = type_start_end.split()
#
#             # Convert start and stop indices to integer
#             start, end = int(start), int(end)
#
#             # Build text bound object
#             assert id not in textbounds
#             textbounds[id] = Textbound(
#                           id = id,
#                           type_= type_,
#                           start = start,
#                           end = end,
#                           text = text,
#                           )
#
#     return textbounds

# def parse_attributes(lines):
#     """
#     Parse attributes, returning a list of Textbound.
#         Assume all attributes are 'Value'
#
#         ex.
#
#         A2      Value T4 current
#         A3      Value T11 none
#
#     """
#
#     attributes = {}
#     for l in lines:
#
#         if ATTRIBUTE_RE.search(l):
#
#             # Split on tabs
#             attr_id, attr_textbound_value = l.split('\t')
#
#             type, tb_id, value = attr_textbound_value.split()
#
#             # Add attribute to dictionary
#             assert tb_id not in attributes
#             attributes[tb_id] = Attribute( \
#                     id = attr_id,
#                     type = type,
#                     textbound = tb_id,
#                     value = value)
#     return attributes

#
# def get_unique_arg(argument, arguments):
#
#     if argument in arguments:
#         argument_strip = argument.rstrip(string.digits)
#         for i in range(1, 20):
#             argument_new = f'{argument_strip}{i}'
#             if argument_new not in arguments:
#                 break
#     else:
#         argument_new = argument
#
#     assert argument_new not in arguments, "Could not modify argument for uniqueness"
#
#     if argument_new != argument:
#         #logging.warn(f"Event decoding: '{argument}' --> '{argument_new}'")
#         pass
#
#     return argument_new
#
# def parse_events(lines):
#     """
#     Parse events, returning a list of Textbound.
#
#     ex.
#         E2      Tobacco:T7 State:T6 Amount:T8 Type:T9 ExposureHistory:T18 QuitHistory:T10
#         E4      Occupation:T9 State:T12 Location:T10 Type:T11
#
#         id     event:tb_id ROLE:TYPE ROLE:TYPE ROLE:TYPE ROLE:TYPE
#     """
#
#     events = {}
#     for l in lines:
#         if EVENT_RE.search(l):
#
#             # Split based on white space
#             entries = [tuple(x.split(':')) for x in l.split()]
#
#             # Get ID
#             id = entries.pop(0)[0]
#
#             # Entity type
#             event_type, _ = tuple(entries[0])
#
#             # Role-type
#             arguments = OrderedDict()
#             for i, (argument, tb) in enumerate(entries):
#
#                 argument = get_unique_arg(argument, arguments)
#                 assert argument not in arguments
#                 arguments[argument] = tb
#
#             # Only include desired arguments
#             events[id] = Event( \
#                       id = id,
#                       type_ = event_type,
#                       arguments = arguments)
#
#     return events
#
#
#
#
# def parse_relations(lines):
#     """
#     Parse events, returning a list of Textbound.
#
#     ex.
#     R1  attr Arg1:T2 Arg2:T1
#     R2  attr Arg1:T5 Arg2:T6
#     R3  attr Arg1:T7 Arg2:T1
#
#     """
#
#     relations = {}
#     for line in lines:
#         if RELATION_RE.search(line):
#
#             # road move trailing white space
#             line = line.rstrip()
#
#             x = line.split()
#             id = x.pop(0)
#             role = x.pop(0)
#             arg1 = x.pop(0).split(':')[1]
#             arg2 = x.pop(0).split(':')[1]
#
#             # Only include desired arguments
#             assert id not in relations
#             relations[id] = Relation( \
#                       id = id,
#                       role = role,
#                       arg1 = arg1,
#                       arg2 = arg2)
#
#     return relations
