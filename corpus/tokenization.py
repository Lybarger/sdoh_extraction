
import re

import spacy

from spacy.tokenizer import Tokenizer
from spacy.attrs import ORTH, NORM
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
import logging
from tqdm import tqdm
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


LANG = 'en_core_web_sm'

DISABLE = ["tagger", "ner", "lemmatizer"]

#INFIXES = ['[~:*\(\)\/-]']
#INFIXES = ['[:\(\)\/-]']
INFIXES = []

#PREFIXES = ["\[\*\*", "~~\*\*\*", "<<"]
#SUFFIXES = ["\*\*\]", "\*\*\*~~", ">>", '([a-zA-Z]{1,2}\.){2,4}']

PREFIXES = []
#SUFFIXES = ['([a-zA-Z]{1,2}\.){2,4}']
SUFFIXES = []

#SPECIAL_CASES = [
#        ("[**", [{ORTH: "[**"}]),
#        ("**]", [{ORTH: "**]"}]),
#        ("~~***", [{ORTH: "~~***"}]),
#        ("***~~", [{ORTH: "***~~"}]),
#]

SPECIAL_CASES = []



ASCII_MAP = {}
# Hyphen
# ASCII_MAP[173] = 45
ASCII_MAP['\xad'] = '-'


'''
This paper provides some spacy customization suggestions:
https://medinform.jmir.org/2020/7/e18417/pdf
See the appendix

Name Description Additional affix rules Text Tokens
Prefixes A regex-based function for
identifying token prefixes
'[**',
'**]',
(\d+/\d+ | \d+\,\d+ | \d*.\d+ | \d+. | \d+),
'-', '.', 'O2', 'o2'
[**2012 [**, 2012
2.3units 2.3, units
Infixes
A regex-based function for
identifying the infixes in a
token
'(', '+', '->', '/', '-', ':' 50+units
lisin/hctz
50, +, units
lisin, /, hctz
Suffixes A regex-based function for
identifying token suffixes
'[**', '**]', 'mg', 'prn', 'qhs', 'hrs', 'O2',
'o2', '(s)', '-', ':', 'NC', 'SQ', 'PRBC',
'QAM', 'QPM', 'PM', 'nc', 'MWF', 'QD',
'RBCs'
20mg
8hrs
20, mg
8, hrs



'''


def line_break_boundary(doc):
    for token in doc[:-1]:
        if token.text.isspace() and ('\n' in token.text):
            doc[token.i+1].is_sent_start = True
    return doc

def get_tokenizer():
    nlp = spacy.load(LANG, disable=DISABLE)
    return nlp


def get_tokenizer_OLD( \
        lang = LANG,
        disable = DISABLE,
        #prefixes_custom = PREFIXES,
        #infixes_custom =INFIXES,
        #suffixes_custom = SUFFIXES,
        #special_cases = None,
        linebreak_bound = False):

    '''
    http://www.longest.io/2018/01/27/spacy-custom-tokenization.html
    '''

    nlp = spacy.load(lang, disable=disable)

    # Incorporate custom prefixes
    #prefixes_default = list(nlp.Defaults.prefixes)
    #prefixes_all = tuple(prefixes_default + prefixes_custom)
    #prefixes_re = spacy.util.compile_prefix_regex(prefixes_all)

    # Incorporate custom infixes
    #infixes_default = list(nlp.Defaults.infixes)
    #infixes_all = tuple(infixes_default + infixes_custom)
    #infixes_re = spacy.util.compile_infix_regex(infixes_all)

    # Incorporate custom suffixes
    #suffixes_default = list(nlp.Defaults.suffixes)
    #suffixes_all = tuple(suffixes_default + suffixes_custom)
    #suffix_re = spacy.util.compile_suffix_regex(suffixes_all)

    # Create tokenizer
    tokenizer = Tokenizer(nlp.vocab,
                    nlp.Defaults.tokenizer_exceptions,
                     #prefix_search = prefixes_re.search,
                     #infix_finditer = infixes_re.finditer,
                     #suffix_search = suffix_re.search,
                     token_match=None)

    # Incorporate special cases
    if special_cases is not None:
        for tok, case in special_cases:
            tokenizer.add_special_case(tok, case)


    nlp.tokenizer = tokenizer

    if linebreak_bound:
        nlp.add_pipe(line_break_boundary, first=True)
    return nlp


def is_ascii(s):

    for c in s:
        if ord(c) > 128:
            logging.warn('{} {} is not ascii'.format(c, ord(c)))
    return True

def map2ascii(text, map=ASCII_MAP):

    for original, new in map.items():
        text = text.replace(original, new)

    return text

def normalize_linebreaks(text):
    text = re.sub('\r\n', '\n', text)
    return text


def has_windows_linebreaks(text):
    return '\r\n' in text


#def char_cleanup(X, map=CHAR_MAP):
    # Y = []
    # for x in X:
    #     x = ord(x)
    #     y = chr(map.get(x, x))
    #     Y.append(y)
    # Y = ''.join(Y)
    # return Y

def simple_tokenization(A, punct=set('''".:,;/()-\'''')):

    B = []
    for a in A:
        if a in punct:
            B.append(' ')
            B.append(a)
            B.append(' ')
        else:
            B.append(a)

    B = ''.join(B).split()

    assert ''.join(''.join(B).split()) == ''.join(A.split())

    return B

def get_char_indices(text, tokens):

    text_orig = text

    i = 0
    indices = []
    for token in tokens:

        # look for matches
        m = re.search(re.escape(token), text)

        # make sure match found
        assert m

        # store indices
        start = m.start(0) + i
        end = m.end(0) + i
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert end > start
        indices.append((start, end))

        # update counter an text
        i = end
        text = text[m.end(0):]

    assert len(indices) == len(tokens), f'{len(indices)} vs {len(tokens)}'

    for j, (start, end) in enumerate(indices):
        assert tokens[j] == text_orig[start:end], f'{repr(tokens[j])} vs {repr(text[start:end])}'

    return indices

def rm_extra_linebreaks(text):

    # get original character count out text without white
    char_count = len(text)
    wo_ws = ''.join(text.split())

    # find all redundant linebreaks
    matches = list(re.finditer('\n[ \n]+', text))

    # iterate over matches
    for m in matches:

        orig = m.group(0)

        start = m.start()
        end = m.end()
        n =  end - start
        new = ' '*(n-1) + '\n'
        assert len(new) == len(orig)

        text = text[:start] + new + text[end:]

    assert char_count == len(text)
    assert wo_ws == ''.join(text.split())

    return text

def rm_footer(text, footer):

    n = len(footer)
    assert text[-n:] == footer, f"{text[-n:]} vs {footer}"

    text = text[:-n]

    return text



def tokenize_document(text, tokenizer, keep_ws=False, max_sent_count=None, pad_start=False, pad_end=False):

    spacy_doc = tokenizer(text)
    sents = list(spacy_doc.sents)

    # Filter sentences
    sents = [sent for sent in sents if keep_ws or (not sent.text.isspace())]

    sentences = []
    sent_offsets = []
    token_offsets = []
    for sent in sents:

        sent_offsets.append((sent.start_char, sent.end_char))
        sentences.append(sent.text)
        token_offsets.append([])

        # filter tokens
        tokens = [token for token in sent if keep_ws or (not token.text.isspace())]

        if pad_start:
            token_offsets[-1].append((-1, -1))

        for token in tokens:
            token_start = token.idx
            token_end = token_start + len(token.text)
            token_offsets[-1].append((token_start, token_end))

        if pad_end:
            token_offsets[-1].append((-1, -1))

    for sent in sentences:
        if not keep_ws:
            assert len(sent) > 0

    if max_sent_count is not None:

        sentences = sentences + ['']*max_sent_count
        sent_offsets = sent_offsets + [(0,0)]*max_sent_count
        token_offsets = token_offsets + [[(0,0)]]*max_sent_count

        sentences = sentences[:max_sent_count]
        sent_offsets = sent_offsets[:max_sent_count]
        token_offsets = token_offsets[:max_sent_count]


    assert len(sentences) == len(sent_offsets)
    assert len(sentences) == len(token_offsets)


    return (sentences, sent_offsets, token_offsets)


def get_tokens(text, tokenizer, keep_ws=False):

    spacy_doc = tokenizer(text)
    sents = list(spacy_doc.sents)

    # Filter sentences
    sents = [s for s in sents if keep_ws or (not s.text.isspace())]

    tokens = []
    for sent in sents:


        # filter tokens
        tokens.append([t for t in sent if keep_ws or (not t.text.isspace())])

    return tokens

#def tokenize_corpus(documents, max_sent_count=None,\
#                    linebreak_bound=True, keep_ws=False,
#                    pad_start=False, pad_end=False):
def tokenize_corpus(documents, max_sent_count=None,\
                    keep_ws=False, pad_start=False, pad_end=False):


    tokenizer = get_tokenizer()

    logging.info("Tokenizing documents...")

    sentences = []
    sent_offsets = []
    token_offsets = []
    pbar = tqdm(total=len(documents))
    for doc in documents:
        sent, sent_off, tok_off = tokenize_document( \
                                        text = doc,
                                        tokenizer = tokenizer,
                                        keep_ws = keep_ws,
                                        max_sent_count = max_sent_count,
                                        pad_start = pad_start,
                                        pad_end = pad_end)

        sentences.append(sent)
        sent_offsets.append(sent_off)
        token_offsets.append(tok_off)

        pbar.update()
    pbar.close()
    return (sentences, sent_offsets, token_offsets)




def remove_white_space_at_ends(text, start, end):

    # leading white space
    n = len(text)
    text = text.lstrip()
    start += n - len(text)

    # trailing white space
    n = len(text)
    text = text.rstrip()
    end -= n - len(text)

    return (text, start, end)



def get_context(spacy_doc, start, end, context_len=1):

    target = None

    sentences = list(spacy_doc.sents)
    sentences = [sent for sent in sentences if (not sent.text.isspace())]

    for i, sent in enumerate(sentences):

        sent_start = sent.start_char
        sent_end = sent.end_char

        start_match = (start >= sent_start) and (start <  sent_end)
        end_match =   (end   >  sent_start) and (end   <= sent_end)

        if start_match:
            target = i
            if not end_match:
                logging.warn(f"get_context: start and end not an same sentence")
            break

    assert target is not None

    target_end = target + 1
    target_start = target_end - context_len

    context = sentences[target_start:target_end]
    context = [sentence.text for sentence in context]
    context = " ".join(context)

    return context
