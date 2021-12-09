
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
