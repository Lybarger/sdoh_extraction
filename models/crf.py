

import torch
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.attention import BilinearAttention
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

from tqdm import tqdm
import numpy as np
import logging
from tqdm import tqdm
import joblib
import math
from collections import OrderedDict

# from corpus.event import BIO_to_span_seq
# from models.utils import create_mask

from models.utils import loss_reduction

import config.constants as C


BIO_DEFAULT = True
PAD_START = True
PAD_END = True


def strip_BIO_tok(x, begin=C.BEGIN, inside=C.INSIDE):

    return re.sub('(^{})|(^{})'.format(begin, inside), '', x)

def is_begin(x):
    return bool(re.match('^{}'.format(C.BEGIN), x))

def is_inside(x):
    return bool(re.match('^{}'.format(C.INSIDE), x))

def is_outside(x):
    return x == C.OUTSIDE



def BIO_to_span_seq(seq, check_count=False, as_Span=True):
    '''

    Finds spans in BIO sequence

    NOTE: start span index is inclusive, end span index is exclusive
            e.g. like Python lists

    '''

    # Not in a span
    in_span = False


    spans = []
    b_count = 0
    start = -1
    end = -1
    active_tag = None
    for i_tok, x in enumerate(seq):

        # Current tag
        tag = strip_BIO_tok(x)

        # Outside token
        if is_outside(x):

            # The span has ended
            if active_tag is not None:
                s = Span( \
                    type_ = active_tag,
                    sent_idx = None,
                    tok_idxs = (start, end))
                spans.append(s)

            # Not in a span
            active_tag = None

        # Span beginning
        elif is_begin(x):

            # The span has ended
            if active_tag is not None:
                s = Span( \
                    type_ = active_tag,
                    sent_idx = None,
                    tok_idxs = (start, end))
                spans.append(s)

            # Update active tag
            active_tag = tag

            # Index of current span start
            start = i_tok
            end = i_tok + 1

            # Increment b count
            b_count += 1

        # Span inside and current tag matches active tag
        # e.g. well-formed span
        elif is_inside(x) and (tag == active_tag):

            end += 1

        # This is an inside span that is ill formed
        else:

            # Append ill formed span
            if active_tag is not None:
                s = Span( \
                    type_ = active_tag,
                    sent_idx = None,
                    tok_idxs = (start, end))
                spans.append(s)
            # Update active tag
            active_tag = tag

            # Index of current span start
            start = i_tok
            end = i_tok

    # Last token might be part of a valid span
    if active_tag is not None:
        s = Span( \
            type_ = active_tag,
            sent_idx = None,
            tok_idxs = (start, end))
        spans.append(s)

    # Get span count
    s_count = len(spans)

    if check_count and (b_count != s_count):
        msg = \
        '''Count mismatch:
        seq = {}
        Begin count = {}
        span count = {}'''.format(seq, b_count, s_count)
        logging.warn(msg)

    if as_Span:
        return spans
    else:
        types_ = [s.type_ for s in spans]
        indices = [list(s.indices) for s in spans]
        return (types_, indices)

def seq_tags_to_spans(y, map_fn):


    # Number of spans per sequence
    count_ = []

    # Span labels (variable length)
    labels = []

    # Indices of spans (variable length)
    indices_tmp = []

    # Loop on sequences
    for seq in y:

        # At least one nonnegative label found
        if any(seq):

            # Map IDs to labels
            labs = [map_fn(s) for s in seq]

            # Get spans as labels in indices
            labs, idx = BIO_to_span_seq(labs, \
                                        end_exclusive = False,
                                        as_Span = False)

            # Number of spans per sequence
            cnt = len(labs)

        # Only negative labels found
        else:
            labs = []
            idx = []
            cnt = 0

        # Append to batch results
        labels.append(labs)
        indices_tmp.append(idx)
        count_.append(cnt)

    # Batch size
    batch_size = len(y)

    # Maximum span count per sequence
    max_span_count = max(max(count_), 1)
    count_ = np.array(count_)

    # Initialize span indices
    indices = np.zeros((batch_size, max_span_count, 2), dtype=np.int32)

    # Initialize span indices mask
    mask = np.zeros((batch_size, max_span_count), dtype=np.int32)

    # Loop on sequences in batch
    assert batch_size > 0
    for i in range(batch_size):

        # Number of spans in sequence
        c = count_[i]

        # Only update if span(s) present
        if c > 0:

            # Update span indices
            indices[i,:c,:] = np.array(indices_tmp[i])

            # Update span mask
            mask[i,:c] = 1

    # Convert to Tensor
    indices = torch.LongTensor(indices)
    mask = torch.LongTensor(mask)

    return (labels, indices, mask)

def multitask_seq_tags_to_spans(y, map_fn, entity):


    labels = {}
    indices = {}
    mask = {}

    for k, labs in y.items():
        labels[k], indices[k], mask[k] = seq_tags_to_spans(labs, map_fn[k][entity])

    return (labels, indices, mask)





def add_bio_labels(seq, start, end, lab, num_tags, \
        pad_start = False,
        neg_label = None):


    assert isinstance(lab, int)

    n = len(seq)

    # Iterate over indices
    for j, tok_idx in enumerate(range(start, end)):

        # Increment index, if padding start
        tok_idx += int(pad_start)

        # Implement BIO
        if j > 0:
            lab_new = lab + (num_tags - 1)
        else:
            lab_new = lab


        if tok_idx < n:
            if neg_label is not None:
                assert seq[tok_idx] in [neg_label, lab_new, lab + (num_tags - 1)], '{} not in {}, offset = {}'.format(seq[tok_idx], [neg_label, lab_new, lab + (num_tags - 1)], (num_tags - 1))

            seq[tok_idx] = lab_new
        else:
            logging.info("Could not include {}th token in seq of length {}".format(j, n))

    return seq



def tag_to_span_lab(seq_label, num_tags_orig):
    '''
    Convert BIO representation to span representation


    Parameters
    ----------
    seq_label: int, span label with BIO indices
    num_tags: int, number of original tags, without BI prefixes
    '''

    is_O = 0
    is_B = 0
    is_I = 0

    # Number of positive (non-negative tags)
    num_pos_tags = num_tags_orig - 1

    # Convert sequence label to span label (i.e. convert BI)
    if seq_label > num_pos_tags:
        span_label = seq_label - num_pos_tags
        is_I = 1
    elif seq_label > 0:
        span_label = seq_label
        is_B = 1
    else:
        span_label = seq_label
        is_O = 1

    return (span_label, is_O, is_B, is_I)





def BIO_to_span(labels, id_to_label=None, lab_is_tuple=True,
                     num_tags_orig=None, tag_to_lab_fn=None):
    '''

    Finds spans in BIO sequence

    NOTE: start span index is inclusive, end span index is exclusive
            e.g. like Python lists

    Parameters
    ----------
    labels: list of token label ids (tag ids)
    '''

    spans = []
    begin_count = 0
    start = -1
    end = -1
    active_tag = None

    # No non-negative labels, so return empty list
    if not any(labels):
        return []

    # Loop on tokens in seq
    for i, lab in enumerate(labels):

        if tag_to_lab_fn is not None:
            tag, is_O, is_B, is_I = tag_to_lab_fn(lab)

        # Label is tuple
        elif lab_is_tuple:
            # Convert current sequence tag label to span label
            if id_to_label is None:
                prefix, tag  = lab
            else:
                prefix, tag  = id_to_label[lab]

            is_O = prefix == C.OUTSIDE
            is_B = prefix == C.BEGIN
            is_I = prefix == C.INSIDE



        # Label is not tuple, so use number of tags to resolve BI
        # prefixes
        else:
            assert num_tags_orig is not None
            tag, is_O, is_B, is_I = tag_to_span_lab(lab, num_tags_orig)

        # Outside label
        if is_O:

            # The span has ended
            if active_tag is not None:
                spans.append((active_tag, start, end))

            # Not in a span
            active_tag = None

        # Span beginning
        elif is_B:

            # The span has ended
            if active_tag is not None:
                spans.append((active_tag, start, end))

            # Update active tag
            active_tag = tag

            # Index of current span start
            start = i
            end = i + 1

            # Increment begin count
            begin_count += 1

        # Span inside and current tag matches active tag
        # e.g. well-formed span
        elif is_I and (tag == active_tag):
            end += 1

        # Ill formed span
        elif is_I and (tag != active_tag):

            # Capture end of valid span
            if active_tag is not None:
                spans.append((active_tag, start, end))

            # Not in a span
            active_tag = None

        else:
            raise ValueError("could not assign label")

    # Last token might be part of a valid span
    if active_tag is not None:
        spans.append((active_tag, start, end))

    # Get span count
    span_count = len(spans)

    if True and (begin_count != span_count):
        msg = \
        '''Count mismatch:
        seq = {}
        Begin count = {}
        span count = {}'''.format(seq, begin_count, span_count)
        logging.warn(msg)

    return spans



class CRF(nn.Module):
    '''
    CRF
    '''
    def __init__(self, num_tags, embed_size,
            constraints = None,
            incl_start_end = True):
        super(CRF, self).__init__()

        self.num_tags = num_tags
        self.constraints = constraints
        self.incl_start_end = incl_start_end

        # Linear projection layer
        self.projection = nn.Linear(embed_size, num_tags)

        # Create event-specific CRF
        self.crf = ConditionalRandomField( \
                        num_tags = num_tags,
                        constraints = constraints,
                        include_start_end_transitions = incl_start_end)

        logging.info('CRF')
        logging.info('\tnum_tags: {}'.format(num_tags))
        logging.info('\tembed_size: {}'.format(embed_size))
        logging.info('\tconstraints: {}'.format(constraints))
        logging.info('\tinclude_start_and_transitions: {}'.format(incl_start_end))
        logging.info('\tprojection: {}'.format(self.projection))


    def forward(self, X, y=None, mask=None, seq_feats=None):
        '''
        Generate predictions
        '''

        # Append sequence features
        if seq_feats is not None:
            batch, seq_len, embed = X.size()
            seq_feats = seq_feats.unsqueeze(1).repeat(1, seq_len, 1)
            X = torch.cat((X, seq_feats), 2)

        # Project to logits
        logits = self.projection(X)


        # Training mode
        if self.training:
            loss = - self.crf( \
                            inputs = logits,
                            tags = y,
                            mask = mask)

            pred = torch.Tensor([0])

        # Evaluation mode
        else:
            loss =  torch.Tensor([0])

            # Predictions
            best_paths = self.crf.viterbi_tags( \
                                            logits = logits,
                                            mask = mask)
            pred, score = zip(*best_paths)
            pred = list(pred)


        return (loss, pred)

class MultitaskCRF(nn.Module):
    '''
    Multitask CRF
    '''
    def __init__(self, event_types, num_tags, embed_size,
            constraints = None,
            incl_start_end = True,
            reduction = 'sum'):
        super(MultitaskCRF, self).__init__()


        self.event_types = event_types
        self.num_tags = num_tags
        self.constraints = constraints
        self.incl_start_end =  incl_start_end
        self.reduction = reduction

        # Initialize dictionaries
        self.crf = nn.ModuleDict({})

        logging.info('')
        logging.info('Multitask CRF')

        # Loop on events
        for t in self.event_types:

            logging.info('')
            logging.info(t)

            # Create event-specific CRF
            self.crf[t] = CRF( \
                                num_tags = num_tags[t],
                                embed_size = embed_size,
                                constraints = constraints,
                                incl_start_end = incl_start_end)

    def forward(self, X, y=None, mask=None, seq_feats=None):
        '''
        Generate predictions
        '''

        # Initialize dictionary of predictions, log likelihood, etc.
        pred = OrderedDict()
        loss = OrderedDict()

        # Loop on events
        for t in self.event_types:

            # Project to logits
            loss[t], pred[t] = self.crf[t]( \
                        X = X,
                        y = None if y is None else y[t],
                        mask = mask,
                        seq_feats = seq_feats)


        # Aggregate loss values
        loss = loss_reduction(loss, self.reduction)

        return (pred, None, loss)

class SpanExtractor(nn.Module):
    '''
    Span extractor
    '''
    def __init__(self, num_tags, embed_size, id_to_label,
            constraints = None,
            incl_start_end = True,
            span_rep = "x,y,x*y",
            ):
        super(SpanExtractor, self).__init__()

        self.num_tags = num_tags
        self.embed_size = embed_size
        self.id_to_label = id_to_label
        self.constraints = constraints
        self.incl_start_end = incl_start_end
        self.span_rep = span_rep

        # Linear projection layer
        self.projection = nn.Linear(embed_size, num_tags)

        # Create event-specific CRF
        self.crf = ConditionalRandomField( \
                        num_tags = num_tags,
                        constraints = constraints,
                        include_start_end_transitions = incl_start_end)

        # Endpoint extractor
        self.endpoint_extractor = EndpointSpanExtractor(embed_size,
                                      combination = span_rep,
                                      num_width_embeddings = None,
                                      span_width_embedding_dim = None,
                                      bucket_widths = False)

        #self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=text_field_embedder.get_output_dim())


        logging.info('SpanExtractor')
        logging.info('\tnum_tags: {}'.format(num_tags))
        logging.info('\tembed_size: {}'.format(embed_size))
        logging.info('\tconstraints: {}'.format(constraints))
        logging.info('\tinclude_start_and_transitions: {}'.format(incl_start_end))
        logging.info('\tprojection: {}'.format(self.projection))
        logging.info('\tendpoint_extractor: {}'.format(self.endpoint_extractor))

    def forward(self, X, y=None, mask=None):
        '''
        Generate predictions
        '''
        # Batch size
        batch_size = batch_size = len(mask)

        # Project to logits
        logits = self.projection(X)

        # Get loss (negative log likely)
        loss = None if y is None else -self.crf( \
                                                inputs = logits,
                                                tags = y,
                                                mask = mask)
        # Best path
        best_paths = self.crf.viterbi_tags( \
                                        logits = logits,
                                        mask = mask)

        # Separate predictions and score
        y_pred, score = zip(*best_paths)
        y_pred = list(y_pred)

        # Get spans from sequence tags
        span_labels, span_indices, span_mask = \
                            seq_tags_to_spans(y_pred, self.id_to_label)

        # Get span representations
        span_embed = self.endpoint_extractor( \
                sequence_tensor = X,
                span_indices = span_indices,
                sequence_mask = mask,
                span_indices_mask = span_mask)

        return (loss, y_pred, span_labels, span_embed)


class MultitaskSpanExtractor(nn.Module):
    '''
    Multitask span extractor
    '''
    def __init__(self, events, entity, num_tags, embed_size, id_to_label,
            constraints = None,
            incl_start_end = True,
            span_rep = "x,y,x*y"):
        super(MultitaskSpanExtractor, self).__init__()


        self.events = events
        self.entity = entity
        self.num_tags = num_tags
        self.embed_size = embed_size
        self.id_to_label = id_to_label
        self.constraints = constraints
        self.incl_start_end = incl_start_end
        self.span_rep = span_rep



        # Initialize dictionaries for CRF
        self.span_extractors = nn.ModuleDict({})

        # Loop on events
        for event in self.events:

            # Create event-specific CRF
            self.span_extractors[event] = SpanExtractor( \
                            num_tags = num_tags[event][entity],
                            embed_size = embed_size,
                            id_to_label = id_to_label[event][entity],
                            constraints = constraints,
                            incl_start_end = incl_start_end,
                            span_rep = span_rep)

            logging.info('')
            logging.info('{}-{}'.format(event, entity))


    def forward(self, X, y=None, mask=None):
        '''
        Generate predictions
        '''
        # Batch size
        batch_size = batch_size = len(mask)

        # Initialize dictionary of predictions, log likelihood, etc.
        loss = {}
        y_pred = {}
        span_labels = {}
        span_embed = {}

        # Loop on events
        for event in self.events:

            # Project to logits
            ls, yp, sl, se = self.span_extractors[event]( \
                        X = X,
                        y = None if y is None else y[event][self.entity],
                        mask = mask)
            loss[event] = ls
            y_pred[event] = yp
            span_labels[event] = sl
            span_embed[event] = se


        return (loss, y_pred, span_labels, span_embed)

#def BIO_to_spanOLD(seq, num_tags):
#    '''
#
#    Finds spans in BIO sequence
#
#    NOTE: start span index is inclusive, end span index is exclusive
#            e.g. like Python lists
#
#    Parameters
#    ----------
#    seq: list of token label ids (tag ids)
#    '''
#
#    spans = []
#    begin_count = 0
#    start = -1
#    end = -1
#    active_tag = None
#
#    # No non-negative labels, so return empty list
#    if not any(seq):
#        return []
#
#    # Loop on tokens in seq
#    for i, x in enumerate(seq):
#
#        # Convert current sequence tag label to span label
#        tag, is_outside, is_begin, is_inside = tag_to_span_lab(x, num_tags)
#
#        # Outside label
#        if is_outside:
#
#            # The span has ended
#            if active_tag is not None:
#                spans.append((active_tag, start, end))
#
#            # Not in a span
#            active_tag = None
#
#        # Span beginning
#        elif is_begin:
#
#            # The span has ended
#            if active_tag is not None:
#                spans.append((active_tag, start, end))
#
#            # Update active tag
#            active_tag = tag
#
#            # Index of current span start
#            start = i
#            end = i + 1
#
#            # Increment begin count
#            begin_count += 1
#
#        # Span inside and current tag matches active tag
#        # e.g. well-formed span
#        elif is_inside and (tag == active_tag):
#            end += 1
#
#        # Ill formed span
#        elif is_inside and (tag != active_tag):
#
#            # Capture end of valid span
#            if active_tag is not None:
#                spans.append((active_tag, start, end))
#
#            # Not in a span
#            active_tag = None
#
#        else:
#            raise ValueError("could not assign label")
#
#    # Last token might be part of a valid span
#    if active_tag is not None:
#        spans.append((active_tag, start, end))
#
#    # Get span count
#    span_count = len(spans)
#
#    if True and (begin_count != span_count):
#        msg = \
#        '''Count mismatch:
#        seq = {}
#        Begin count = {}
#        span count = {}'''.format(seq, begin_count, span_count)
#        logging.warn(msg)
#
#    return spans
