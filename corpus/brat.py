
import re
import glob
import os
import numpy as np
import logging
from collections import Counter
from collections import OrderedDict
from pathlib import Path
import stat
import copy
import re
from tqdm import tqdm
import string

from config.constants import TEXT_FILE_EXT, ANN_FILE_EXT, TRIGGER, ENCODING





COMMENT_RE = re.compile(r'^#')
TEXTBOUND_RE = re.compile(r'^T\d+')
EVENT_RE = re.compile(r'^E\d+\t')
ATTRIBUTE_RE = re.compile(r'^A\d+\t')
RELATION_RE = re.compile(r'^R\d+\t')

TEXTBOUND_LB_SEP = ';'

class Attribute(object):
    '''
    Container for attribute

    annotation file examples:
        A1      Value T2 current
        A2      Value T8 current
        A3      Value T9 none
        A4      Value T13 current
        A5      Value T17 current
    '''
    def __init__(self, id, type_, textbound, value):
        self.id = id
        self.type_ = type_
        self.textbound = textbound
        self.value = value

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return (self.type_ == other.type_) and \
               (self.textbound == other.textbound) and \
               (self.value == other.value)
    def brat_str(self):
        return attr_str(attr_id=self.id, arg_type=self.type_, \
                            tb_id=self.textbound, value=self.value)

    def id_numerical(self):
        assert self.id[0] == 'A'
        id = int(self.id[1:])
        return id


class Textbound(object):
    '''
    Container for textbound

    Annotation file examples:
        T2	Tobacco 30 36	smoker
        T4	Status 38 46	Does not
        T5	Alcohol 47 62	consume alcohol
        T6	Status 64 74	No history
    '''
    def __init__(self, id, type_, start, end, text, tokens=None):
        self.id = id
        self.type_ = type_
        self.start = start
        self.end = end
        self.text = text
        self.tokens = tokens

    def __str__(self):
        return str(self.__dict__)

    def token_indices(self, char_indices):
        i_sent, (out_start, out_stop) = find_span(char_indices, self.start, self.end)
        return (i_sent, (out_start, out_stop))

    def brat_str(self):
        return textbound_str(id=self.id, type_=self.type_, start=self.start, \
                                                end=self.end, text=self.text)

    def id_numerical(self):
        assert self.id[0] == 'T'
        id = int(self.id[1:])
        return id


class Event(object):
    '''
    Container for event

    Annotation file examples:
        E3      Family:T7 Amount:T8 Type:T9
        E4      Tobacco:T11 State:T10
        E2      Alcohol:T5 State:T4

        id     event:head (entities)
    '''

    def __init__(self, id, type_, arguments):
        self.id = id
        self.type_ = type_
        self.arguments = arguments

    def get_trigger(self):
        for argument, tb in self.arguments.items():
            return (argument, tb)

    def __str__(self):
        return str(self.__dict__)

    def brat_str(self, tb_ids_keep=None):
        return event_str(id=self.id, \
                            event_type = self.type_, \
                            textbounds = self.arguments,
                            tb_ids_keep = tb_ids_keep)


class Relation(object):
    '''
    Container for event

    Annotation file examples:
    R1  attr Arg1:T2 Arg2:T1
    R2  attr Arg1:T5 Arg2:T6
    R3  attr Arg1:T7 Arg2:T1

    '''

    def __init__(self, id, role, arg1, arg2):
        self.id = id
        self.role = role
        self.arg1 = arg1
        self.arg2 = arg2

    def __str__(self):
        return str(self.__dict__)

    def brat_str(self):
        return relation_str(id=self.id, role=self.role, \
                            arg1=self.arg1, arg2=self.arg2)


def get_annotations(ann):
    '''
    Load annotations, including taxbounds, attributes, and events

    ann is a string
    '''

    # Parse string into nonblank lines
    lines = [l for l in ann.split('\n') if len(l) > 0]


    # Confirm all lines consumed
    remaining = [l for l in lines if not \
            ( \
                COMMENT_RE.search(l) or \
                TEXTBOUND_RE.search(l) or \
                EVENT_RE.search(l) or \
                RELATION_RE.search(l) or \
                ATTRIBUTE_RE.search(l)
            )
        ]
    msg = 'Could not match all annotation lines: {}'.format(remaining)
    assert len(remaining)==0, msg

    # Get events
    events = parse_events(lines)

    # Get relations
    relations = parse_relations(lines)

    # Get text bounds
    textbounds = parse_textbounds(lines)

    # Get attributes
    attributes = parse_attributes(lines)

    return (events, relations, textbounds, attributes)


def parse_textbounds(lines):
    """
    Parse textbound annotations in input, returning a list of
    Textbound.

    ex.
        T1	Status 21 29	does not
        T1	Status 27 30	non
        T8	Drug 91 99	drug use

    """

    textbounds = {}
    for l in lines:
        if TEXTBOUND_RE.search(l):

            # Split line
            id, type_start_end, text = l.split('\t')

            # Check to see if text bound spans multiple sentences
            mult_sent = len(type_start_end.split(';')) > 1

            # Multiple sentence span, only use portion from first sentence
            if mult_sent:

                # type_start_end = 'Drug 99 111;112 123'

                # type_start_end = ['Drug', '99', '111;112', '123']
                type_start_end = type_start_end.split()

                # type = 'Drug'
                # start_end = ['99', '111;112', '123']
                type_ = type_start_end[0]
                start_end = type_start_end[1:]

                # start_end = '99 111;112 123'
                start_end = ' '.join(start_end)

                # start_ends = ['99 111', '112 123']
                start_ends = start_end.split(';')

                # start_ends = [('99', '111'), ('112', '123')]
                start_ends = [tuple(start_end.split()) for start_end in start_ends]

                # start_ends = [(99, 111), (112, 123)]
                start_ends = [(int(start), int(end)) for (start, end) in start_ends]

                start = start_ends[0][0]

                # ends = [111, 123]
                ends = [end for (start, end) in start_ends]

                text = list(text)
                for end in ends[:-1]:
                    n = end - start
                    assert text[n].isspace()
                    text[n] = '\n'
                text = ''.join(text)

                start = start_ends[0][0]
                end = start_ends[-1][-1]

            else:
                # Split type and offsets
                type_, start, end = type_start_end.split()

            # Convert start and stop indices to integer
            start, end = int(start), int(end)

            # Build text bound object
            assert id not in textbounds
            textbounds[id] = Textbound(
                          id = id,
                          type_= type_,
                          start = start,
                          end = end,
                          text = text,
                          )

    return textbounds


def parse_attributes(lines):
    """
    Parse attributes, returning a list of Textbound.
        Assume all attributes are 'Value'

        ex.

        A2      Value T4 current
        A3      Value T11 none

    """

    attributes = {}
    for l in lines:

        if ATTRIBUTE_RE.search(l):

            # Split on tabs
            attr_id, attr_textbound_value = l.split('\t')

            type, tb_id, value = attr_textbound_value.split()

            attr_ob = Attribute( \
                    id = attr_id,
                    type_ = type,
                    textbound = tb_id,
                    value = value)


            if tb_id in attributes:
                # attribute defined for textbound already, but value and type match
                # ok
                if (attributes[tb_id].type_ == attr_ob.type_) and \
                   (attributes[tb_id].value == attr_ob.value):
                   pass

                # attribute defined for text found already an there's a conflict between the new and old value
                # raise error
                else:
                    logging.warn("Attribute already exists for textbound")
                    logging.warn(f"Existing attribute: {attributes[tb_id]}")
                    logging.warn(f"New attribute:      {attr_ob}")
                    raise ValueError(f"Attribute already defined for textbound")

            # Add attribute to dictionary
            attributes[tb_id] = attr_ob

    return attributes



def get_unique_arg(argument, arguments):

    if argument in arguments:
        argument_strip = argument.rstrip(string.digits)
        for i in range(1, 20):
            argument_new = f'{argument_strip}{i}'
            if argument_new not in arguments:
                break
    else:
        argument_new = argument

    assert argument_new not in arguments, "Could not modify argument for uniqueness"

    if argument_new != argument:
        #logging.warn(f"Event decoding: '{argument}' --> '{argument_new}'")
        pass

    return argument_new

def parse_events(lines):
    """
    Parse events, returning a list of Textbound.

    ex.
        E2      Tobacco:T7 State:T6 Amount:T8 Type:T9 ExposureHistory:T18 QuitHistory:T10
        E4      Occupation:T9 State:T12 Location:T10 Type:T11

        id     event:tb_id ROLE:TYPE ROLE:TYPE ROLE:TYPE ROLE:TYPE
    """

    events = {}
    for l in lines:
        if EVENT_RE.search(l):

            # Split based on white space
            entries = [tuple(x.split(':')) for x in l.split()]

            # Get ID
            id = entries.pop(0)[0]

            # Entity type
            event_type, _ = tuple(entries[0])

            # Role-type
            arguments = OrderedDict()
            for i, (argument, tb) in enumerate(entries):

                argument = get_unique_arg(argument, arguments)
                assert argument not in arguments
                arguments[argument] = tb

            # Only include desired arguments
            events[id] = Event( \
                      id = id,
                      type_ = event_type,
                      arguments = arguments)

    return events


def parse_relations(lines):
    """
    Parse events, returning a list of Textbound.

    ex.
    R1  attr Arg1:T2 Arg2:T1
    R2  attr Arg1:T5 Arg2:T6
    R3  attr Arg1:T7 Arg2:T1

    """

    relations = {}
    for line in lines:
        if RELATION_RE.search(line):

            # road move trailing white space
            line = line.rstrip()

            x = line.split()
            id = x.pop(0)
            role = x.pop(0)
            arg1 = x.pop(0).split(':')[1]
            arg2 = x.pop(0).split(':')[1]

            # Only include desired arguments
            assert id not in relations
            relations[id] = Relation( \
                      id = id,
                      role = role,
                      arg1 = arg1,
                      arg2 = arg2)

    return relations


def get_filename(path):
    root, ext = os.path.splitext(path)
    return root

def filename_check(fn1, fn2):
    '''
    Confirm filenames, regardless of directory or extension, match
    '''
    fn1 = get_filename(fn1)
    fn2 = get_filename(fn2)

    return fn1==fn2


def get_files(path, ext='.', relative=False):
    files = list(Path(path).glob('**/*.{}'.format(ext)))

    if relative:
        files = [os.path.relpath(f, path) for f in files]

    return files

def get_brat_files(path):
    '''
    Find text and annotation files
    '''
    # Text and annotation files
    text_files = get_files(path, TEXT_FILE_EXT, relative=False)
    ann_files = get_files(path, ANN_FILE_EXT, relative=False)

    # Check number of text and annotation files
    msg = 'Number of text and annotation files do not match'
    assert len(text_files) == len(ann_files), msg

    # Sort files
    text_files.sort()
    ann_files.sort()

    # Check the text and annotation filenames
    mismatches = [str((t, a)) for t, a in zip(text_files, ann_files) \
                                           if not filename_check(t, a)]
    fn_check = len(mismatches) == 0
    assert fn_check, '''txt and ann filenames do not match:\n{}'''. \
                        format("\n".join(mismatches))

    return (text_files, ann_files)


# def import_brat_dir(path, sample_count=None):
#
#     text_files, ann_files = get_brat_files(path)
#     file_list = list(zip(text_files, ann_files))
#     file_list.sort(key=lambda x: x[1])
#
#     if (sample_count is not None) and (sample_count != -1):
#         file_list = file_list[0:sample_count]
#         logging.warn(f"-"*80)
#         logging.warn(f"Truncating loaded files to first {sample_count} files")
#         logging.warn(f"-"*80)
#
#     logging.info(f"")
#     logging.info(f"Importing brat directory: {path}")
#     pbar = tqdm(total=len(file_list))
#
#     # Loop on annotated files
#     out = []
#     for fn_txt, fn_ann in file_list:
#
#         # Use filename as ID
#         id = os.path.splitext(os.path.relpath(fn_txt, path))[0]
#
#         # Read text file
#         with open(fn_txt, 'r', encoding=ENCODING) as f:
#             text = f.read()
#
#         # Read annotation file
#         with open(fn_ann, 'r', encoding=ENCODING) as f:
#             ann = f.read()
#
#         out.append((id, text, ann))
#
#         pbar.update(1)
#     pbar.close()
#
#     return out

def mapper(event_type, span_type, export_map):

    if (export_map is not None) and \
       (event_type in export_map) and \
       (span_type in export_map[event_type]):
        span_type = export_map[event_type][span_type]

    return span_type

def textbound_str(id, type_, start, end, text):
    '''
    Create textbounds during from span

    Parameters
    ----------
    id: current textbound id as string
    span: Span object

    Returns
    -------
    BRAT representation of text bound as string
    '''

    if '\n' in text:
        i = 0
        substrings = text.split('\n')
        indices = []
        for s in substrings:
            n = len(s)
            idx = '{start} {end}'.format(start=start + i, end=start + i + n)
            indices.append(idx)
            i += n + 1

        indices = TEXTBOUND_LB_SEP.join(indices)

    else:
        indices = '{start} {end}'.format(start=start, end=end)

    text = re.sub('\n', ' ', text)


    if isinstance(id, str) and (id[0] == "T"):
        id = id[1:]

    return 'T{id}\t{type_} {indices}\t{text}'.format( \
        id = id,
        type_ = type_,
        indices = indices,
        text = text)

    #return 'T{id}\t{type_} {start} {end}\t{text}'.format( \
    #    id = id,
    #    type_ = type_,
    #    start = start,
    #    end = end,
    #    text = text)


def attr_str(attr_id, arg_type, tb_id, value):
    '''
    Create attribute string
    '''

    if isinstance(attr_id, str) and (attr_id[0] == "A"):
        attr_id = attr_id[1:]

    if isinstance(tb_id, str) and (tb_id[0] == "T"):
        tb_id = tb_id[1:]

    return 'A{attr_id}\t{arg_type} T{tb_id} {value}'.format( \
        attr_id = attr_id,
        arg_type = arg_type,
        tb_id = tb_id,
        value = value)


def event_str(id, event_type, textbounds, tb_ids_keep=None):
    '''
    Create event string

    Parameters:
    -----------
    id: current event ID as string
    event_type: event type as string
    textbounds: list of tuple, [(span.type_, id), ...]

    '''

    if isinstance(id, str) and (id[0] == "E"):
        id = id[1:]

    if tb_ids_keep is not None:
        textbounds = OrderedDict([(arg_type, tb_id) \
            for arg_type, tb_id in textbounds.items() if tb_id in tb_ids_keep])

    if len(textbounds) == 0:
        return ''

    else:
        # Start event string
        out = 'E{}\t'.format(id)

        # Create event representation
        event_args = []
        for arg_type, tb_id in textbounds.items():

            if tb_id[0] == "T":
                tb_id = tb_id[1:]

            if arg_type == TRIGGER:
                arg_type = event_type

            out += '{}:T{} '.format(arg_type, tb_id)

        return out


def relation_str(id, role, arg1, arg2):
    '''
    Create event string

    Parameters:
    -----------

    R1	attr Arg1:T2 Arg2:T1

    '''


    if isinstance(id, str) and (id[0] == "R"):
        id = id[1:]

    # Start event string
    out = f'R{id}\t{role} Arg1:{arg1} Arg2:{arg2}'

    return out




def write_file(path, id, content, ext):

    # Output file name
    fn = os.path.join(path, '{}.{}'.format(id, ext))

    # Directory, including path in id
    dir_ = os.path.dirname(fn)
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    # Write file
    with open(fn, 'w', encoding=ENCODING) as f:
        f.write(content)


    #os.chmod(dir_, stat.S_IWGRP)
    #os.chmod(fn, stat.S_IWGRP)
    return fn

# def write_txt(path, id, text, strip_ws=False):
#     '''
#     Write text file
#     '''
#
#     if strip_ws:
#         text = '\n'.join([line.strip() for line in text.splitlines()])
#
#     fn = write_file(path, id, text, TEXT_FILE_EXT)
#     return fn

# def convert(match_obj):
#     if match_obj.group() is not None:
#         return match_obj.group().replace('\n',' ')


def write_txt(path, id, text, fix_linebreak=False, strip_ws=False):
    '''
    Write text file
    '''

    fix_flag=0
    if fix_linebreak:
        max_len = 0
        sentence_list = text.split('\n')

        for sentence in sentence_list:
            if len(sentence)>max_len:
                max_len=len(sentence)

        # Consider only reports where the maximum char length for each sentence is shorter than 100
        if max_len<=100:
            fix_flag=1

    if fix_flag==1:
        # Find indices where the extraneous linebreak should be fixed
        updated_indices = []
        for match in re.finditer(r'[A-Za-z,] *\n *[A-Za-z0-9]', text):
            index = match.start()+match.group().find('\n')
            indices.append(index)

            # updated_text = re.sub(r'[A-Za-z,] *\n *[A-Za-z0-9]', convert, text)

        # Among found indices, remove those who are followed by ':' in the next sentence
        updated_indices_2 = []
        for updated_index in updated_indices:
            if ':' in text[updated_index:].split('\n')[1]:
                pass
            else:
                updated_indices_2.append(updated_index)

        updated_text = ''
        if len(updated_indices_2)==0:
            updated_text = text
        else:
            for char_idx in range(len(text)):
                if char_idx in updated_indices_2:
                    updated_text+=' '
                else:
                    updated_text+=text[char_idx]

        text = updated_text

    if strip_ws:
        text = '\n'.join([line.strip() for line in text.splitlines()])

    fn = write_file(path, id, text, TEXT_FILE_EXT)
    return fn

def write_ann(path, id, ann):
    '''
    Write annotation file
    '''
    fn = write_file(path, id, ann, ANN_FILE_EXT)
    return fn

def get_max_id(object_dict):
    x = [v.id_numerical() for k, v in object_dict.items()]
    if len(x) == 0:
        i = 0
    else:
        i = max(x)
    return i


def get_next_index(d):
    ids = [x.id for x in d.values()]

    if len(ids) == 0:
        last = 0
    else:
        ids = [int(id[1:]) for id in ids]
        last = max(ids)

    return last + 1
