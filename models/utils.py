

from allennlp.nn import util
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, Counter
import pandas as pd
from torch.nn.modules.activation import Sigmoid, ReLU, Tanh
import logging
from tensorboardX import SummaryWriter
from datetime import datetime

import subprocess
# import GPUtil

def nested_dict_to_list(D):

    # Check dimensions
    len_dict = {evt: {ent: len(labs) for ent, labs in ents.items()} \
                                             for evt, ents in D.items()}


    lengths = [ln for evt, ents in len_dict.items() \
                                          for ent, ln in ents.items()]
    assert len(set(lengths)) == 1, "length mismatch: {}".format(len_dict)
    length = lengths[0]

    # Build list of dictionaries
    L = []
    for i in range(length):

        # Loop on dictionary
        d = {evt:{ent:labs[i] for ent, labs in ents.items()} \
                                          for evt, ents in D.items()}

        # Append to list
        L.append(d)
    return L


def get_freer_gpu(verbose=True):


    try:
        subprocess.call(['nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp'])
        with open('tmp', 'r') as f:
            X = f.readlines()
        if verbose:
            logging.info('\n'.join(X))
        memory_available = [int(x.split()[2]) for x in X]
        return np.argmax(memory_available)

    except OSError:
        return 0



def _create_mask(seq_len, max_len):
    '''
    Create sequence mask
    '''
    mask = np.zeros(max_len, dtype=np.int32)
    mask[:seq_len] = 1
    return mask

def create_mask(seq_lens, max_len):

    # Single sequence mask
    if isinstance(seq_lens, (int, float)):
        return _create_mask(seq_lens, max_len)

    # Sequence of sequence masks
    elif isinstance(seq_lens, (np.ndarray, np.generic, list)):
        return np.array([_create_mask(s, max_len) for s in seq_lens])

    # Error
    else:
        raise TypeError("{} invalid type".format(type(seq_lens)))

def trig_exp(x, num_entity):
    '''
    Function for expanding dimensions and tiling
    # (batch_size, num_trig) --> (batch_size, num_trig, num_entity)
    '''
    return x.unsqueeze(2).expand(-1, -1, num_entity)

def entity_exp(x, num_trig):
    '''
    Function for expanding dimensions and tiling
    # (batch_size, num_entity) --> (batch_size, num_trig, num_entity)
    '''
    return x.unsqueeze(1).expand(-1, num_trig, -1)


def one_hot(X, num_tags, low_val=0, high_val=1):
    '''
    Convert last dimension to one hot encoding
    '''

    # Get device
    device = X.device

    # Input dimensions
    dims = tuple(X.shape)

    # Initialize output
    dims_tags = (*dims, num_tags)
    #Y = torch.Tensor(*dims_tags, device=device)).type(X.type()).fill_(low_val)
    Y = torch.Tensor(*dims_tags).to(device).type(X.type()).fill_(low_val)

    # Convert to one hot encoding
    dims_exp = (*dims, 1)
    last_dim = len(dims_exp) - 1
    Y.scatter_(last_dim, X.view(*dims_exp), high_val)

    return Y


def get_predictions(scores, mask):
    '''
    Get predictions from logits and apply mask

    Parameters
    ----------
    scores: (batch_size, num_trig, num_entity, num_tags) OR
                        (batch_size, num_entity, num_tags)
    mask: (batch_size, num_trig, num_entity) OR
                        (batch_size, num_entity)
    '''

    # Get predictions from label scores
    # (batch_size, num_trig, num_entity) OR (batch_size, num_entity)
    max_scores, label_indices = scores.max(-1)

    # For masked spans to NULL label (i.e. 0)
    labels_masked = label_indices.to(mask.device)*mask

    return labels_masked

def get_argument_pred(span_pred, arg_scores, arg_mask, is_req=False):
    '''

    Parameters
    ----------
    span_pred: span label predictions (NOT SCORES) of size
                        (batch_size, num_arg)
    arg_scores: argument scores of size
                        (batch_size, num_trig, num_arg, num_tags)
    arg_mask: argument mask
                        (batch_size, num_trig, num_arg)

    Returns
    -------
    arg_span_pred: highest probability argument for each trigger (event)
                if as_one_hot:
                    size is (batch_size, num_trig, num_arg)
                else:
                    size is (batch_size, num_trig)
    '''

    # Get dimensionality
    batch_size, num_trig, num_arg, num_tags = tuple(arg_scores.shape)
    assert num_tags == 2, 'binary assumption incorrect'

    # Get device
    device = span_pred.device

    # Get span label scores
    # (batch_size, num_arg)
    span_pred_mask = (span_pred > 0).type(torch.LongTensor).to(device)
    # (batch_size, num_trig, num_arg)
    span_pred_mask = entity_exp(span_pred_mask, num_trig)

    # Mask
    # (batch_size, num_trig, num_arg)
    mask = (arg_mask*span_pred_mask).bool().to(device)

    # Argument is required, so find 1 highest probability argument
    if is_req:

        # Convert scores to probs
        # (batch_size, num_trig, num_arg, 2)
        arg_prob = F.softmax(arg_scores, dim=3)

        # Get positive arg label scores
        # (batch_size, num_trig, num_arg)
        arg_probs_pos = arg_prob[:, :, :, 1]

        # Force invalid values to 0
        # (batch_size, num_trig, num_arg)
        arg_probs_pos[~mask] = 0

        # Get highest probability argument
        # (batch_size, num_trig)
        _, arg_pred = arg_probs_pos.max(-1)

        # Convert to mask as one-hot encoding
        # (batch_size, num_trig, num_arg)
        arg_pred = one_hot(arg_pred, num_arg, low_val=0, high_val=1)

        # Avoid assigning negative labels (mask labels)
        arg_pred[~mask] = 0

    # Argument is optional, so 0 or more arguments are valid
    else:
        # Get masked predictions
        # (batch_size, num_trig, num_arg)
        arg_pred = get_predictions(arg_scores, mask)


    return arg_pred


def map_dict_builder(mapping):
    '''
    Create mapping dictionary from list of labels
    '''
    to_id = OrderedDict((x, i) for i, x in enumerate(mapping))
    from_id = OrderedDict((i, x) for i, x in enumerate(mapping))

    return (to_id, from_id)


def get_mapping_dicts(mapping):
    '''
    Get mapping dictionaries from list of labels
    '''

    # Create mapping dictionaries
    # Loop on determinants
    to_id = {}
    to_lab = {}
    for evt_typ, map_ in mapping.items():
        to_id[evt_typ], to_lab[evt_typ] = map_dict_builder(map_)

    #logging.info("Label to ID mapping functions:")
    #for K, V in to_id.items():
    #    logging.info('\t{}'.format(K))
    #    for k, v in V.items():
    #        logging.info('\t\t{} --> {}'.format(k, v))

    #logging.info("ID to Label mapping functions:")
    #for K, V in to_lab.items():
    #    logging.info('\t{}'.format(K))
    #    for k, v in V.items():
    #        logging.info('\t\t{} --> {}'.format(k, v))


    return (to_id, to_lab)

def get_num_tags_dict(mapping):
    '''
    Get tagged count as dictionary of int
    '''

    # Create mapping dictionaries
    # Loop on determinants
    num_tags = OrderedDict()
    for evt_typ, map_ in mapping.items():
        num_tags[evt_typ] = len(map_)

    return num_tags

def get_nested_label_map(label_map):


    # Initialize output
    to_id = OrderedDict()
    to_lab = OrderedDict()


    # Loop on event types
    for evt_typ, arguments in label_map.items():
        to_id[evt_typ] = OrderedDict()
        to_lab[evt_typ] = OrderedDict()

        # Loop on argument types
        for arg_typ, map_ in arguments.items():
            to_id[evt_typ][arg_typ], to_lab[evt_typ][arg_typ] = \
                                                  map_dict_builder(map_)

    logging.info("Label to ID mapping functions:")
    for evt_typ, arguments in to_id.items():
        logging.info('\t{}'.format(evt_typ))
        for arg_typ, map_ in arguments.items():
            logging.info('\t\t{}'.format(arg_typ))
            for k, v in map_.items():
                logging.info('\t\t\t{} --> {}'.format(k, v))

    logging.info("ID to Label mapping functions:")
    for evt_typ, arguments in to_lab.items():
        logging.info('\t{}'.format(evt_typ))
        for arg_typ, map_ in arguments.items():
            logging.info('\t\t{}'.format(arg_typ))
            for k, v in map_.items():
                logging.info('\t\t\t{} --> {}'.format(k, v))

    return (to_id, to_lab)

def get_nested_num_tags(label_map):

    num_tags = OrderedDict()
    for evt_typ, arguments in label_map.items():
        num_tags[evt_typ] = OrderedDict()
        for arg_typ, map_ in arguments.items():
            num_tags[evt_typ][arg_typ] = len(map_)

    logging.info("Number of tags by event-argument:")
    for evt_typ, arguments in num_tags.items():
        logging.info('\t{}'.format(evt_typ))
        for arg_typ, cnt in arguments.items():
            logging.info('\t\t{} = {}'.format(arg_typ, cnt))

    return num_tags

def get_activation_fn(activation):

    if activation == 'sigmoid':
        return Sigmoid()
    elif activation == 'relu':
        return ReLU()
    elif activation == 'tanh':
        return Tanh()
    else:
        raise ValueError("incorrect activation: {}".format(activation))

# def get_device(device_id=None):
#     '''
#     Determine if GPU (CUDA) is functional
#     '''
#
#     logging.info('Get device')
#     logging.info('\tInput device id:\t{}'.format(device_id))
#
#     # Get GPU count
#     gpu_dev_count = torch.cuda.device_count()
#     logging.info("\tCUDA device count:\t{}".format(gpu_dev_count))
#
#     # No GPU found
#     if (gpu_dev_count == 0) or (device_id == -1) or (device_id is None):
#
#         if not ((device_id is None) or (device_id == -1)):
#             logging.warn('Input device_id provided, but could not find any GPU. Defaulting to CPU.')
#
#         device = 'cpu'
#
#     # At least 1 GPU found
#     elif gpu_dev_count > 0:
#
#         # Print GPU utilization
#         logging.info('\tGPU utilization:\n{}'.format(GPUtil.showUtilization()))
#
#         # Use provided GPU ID
#         if device_id is not None:
#             device = 'cuda:{}'.format(device_id)
#
#         # Determine GPU ID, based on availability
#         else:
#             device_id = GPUtil.getAvailable( \
#                                     order = 'memory',
#                                     limit = 1,
#                                     maxLoad = 0.3,
#                                     maxMemory = 0.4,
#                                     includeNan=False,
#                                     excludeID=[],
#                                     excludeUUID=[])
#             assert len(device_id) == 1
#             device_id = device_id[0]
#
#             device = 'cuda:{}'.format(device_id)
#     else:
#         raise ValueError("invalid GPU device count")
#
#     logging.info("\tOutput device:\t{}".format(device))
#
#     return device


def get_deviceOLD():
    '''
    Determine if GPU (CUDA) is functional
    '''

    #if torch.cuda.is_available():
    #    device = 'cuda'
    #else:
    #    device = 'cpu'

    logging.info('Get device')

    gpu_dev_count = torch.cuda.device_count()
    logging.info("\tCUDA device count:\t{}".format(gpu_dev_count))

    if gpu_dev_count == 0:
        device = 'cpu'
    elif gpu_dev_count > 0:
        free_gpu = get_freer_gpu()
        logging.info("\tFreer device:\t{}".format(free_gpu))
        device = 'cuda:{}'.format(free_gpu)
    else:
        raise ValueError("invalid GPU device count")
    #device = 'cuda'
    #try:
    #    x = torch.ones(1).to(device)*torch.ones(1).to(device)
    #except:
    #    device = 'cpu'

    logging.info("\tDevice:\t{}".format(device))


    return device

def pad1D(X, max_len, dtype=None):
    '''
    Pad 1 dimensional vector
    '''
    if len(X) == 0:
        X = [0]

    X = np.array(X)

    # Initialize array of zeros with desired size
    if dtype == None:
        dtype = X.dtype
    Y = np.zeros(max_len, dtype=dtype)

    # Insert variable length array into fixed/padded array
    r = X.shape[0]
    Y[:r] = X

    return Y
def mem_size(x, units='MB'):
    size = x.element_size()*x.nelement()
    if units == 'MB':
        return '{:.1f} MB'.format(size/1e6)
    if units == 'GB':
        return '{:.1f} GB'.format(size/1e9)

def tensor_summary(tensor, name=None):

    if name is None:
        return 'size={},\tmem={}'.format( \
            tensor.size(), mem_size(tensor))
    else:
        return '{}:\t size={},\tmem={}'.format( \
            name, tensor.size(), mem_size(tensor))


def pad_sequences(X, seq_length, fill=0):
    '''


    Parameters
    ----------
    X: list of variable length sequences, list of lists
    seq_length: size of dim 1


    Returns
    -------
    Y: 2D numpy array of shape (seq_count, seq_length)
    '''
    # Get first dimension
    seq_count = len(X)

    # Initialize output
    dtype = np.array([x_ for x in X for x_ in x]).dtype
    Y = np.zeros((seq_count, seq_length), dtype=dtype)
    if fill != 0:
        Y.fill(fill)

    # Iterate over sequences
    for i, x in enumerate(X):
        n = min(len(x), seq_length)
        Y[i,:n] = np.array(x)[:n]

    return Y

def pad_embedding_seq(X, seq_length, fill=0):
    '''


    Parameters
    ----------
    X: sequence of embeddings, list of lists OR list of array OR 2D array
    seq_length: size of dim 0


    Returns
    -------
    Y: 2D numpy array of shape (seq_length, embed_dim)
    '''

    assert len(set([len(x) for x in X])) == 1

    # Convert to numpy array
    X = np.array(X)
    sl, embed_dim = X.shape

    # Initialize output
    dtype = X.dtype
    Y = np.zeros((seq_length, embed_dim), dtype=dtype)
    if fill != 0:
        Y.fill(fill)

    if sl > seq_length:
        X = X[:seq_length,:]

    # Insert into padded array
    Y[:sl,:] = X[:sl,:]

    return Y

def pad2D(X, max_len, dtype=None, fill=0):

    if dtype is None:
        dtype = type(X[0][0])

    Y = []
    for x in X:
        x =sdfssdf
        # Initialize array of fill with desired size
        padded = np.zeros(max_len, dtype=dtype)
        if fill != 0:
            padded.fill(fill)

        # Insert variable length array into fixed/padded array
        x = np.array(x, dtype=dtype)
        r = x.shape[0]

        padded[:r] = x
        Y.append(padded)

    # Concatenate along new dimension
    Y = np.stack(Y, axis=0)

    return Y


def get_num_tags(mapping):

    num_tags = {}
    for k, v in mapping.items():
        num_tags[k] = len(v)


    logging.info("Tag counts:")
    for K, V in num_tags.items():
        logging.info('\t{} = {}'.format(K, V))

    return num_tags


def to_device(X, y, mask, device):

    X = X.to(device)
    y_ = {k: v.to(device) for k, v in y.items()}
    mask = mask.to(device)

    return (X, y_, mask)

def get_dist(X, name='length', bin_size=None):

    # Bin values
    if bin_size is not None:
        X = [int(round(float(x)/bin_size, 0))*bin_size for x in X]

    # Get counts
    counts = Counter(X)

    # Create data frame from counts
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df = df.rename(columns={'index':name, 0:'count'})

    # Sort by first column
    df.sort_values(name, inplace=True)

    return df

def get_len_dist(X, name='length', bin_size=None):

    lengths = [len(x) for x in X]
    return get_dist(lengths, name, bin_size)


def trunc_seqs(X, max_len, bin_size=10):
    '''
    Truncate a list of sequences
    '''

    # Original sequence length distribution
    df_orig = get_len_dist(X, bin_size=10)

    # Truncate sequences
    Y = []
    trunc_count = 0
    for x in X:
        if len(x) > max_len:
            trunc_count += 1
        Y.append(x[0:max_len])

    # Truncated sequence length distribution
    df_trunc = get_len_dist(Y, bin_size=10)

    logging.info("\n")
    logging.info("TRUNCATE SEQUENCES")
    logging.info("Truncated sequence count:\t{}".format(trunc_count))
    logging.info("Original distribution:\n{}".format(df_orig))
    logging.info("Truncated distribution:\n{}".format(df_trunc))


    return Y


def seq_label(current, new_):
    '''
    Get sequence label index (i.e. sentence label) from two different
    label indices
    '''
    # Select highest index)
    return max(current, new_)











def create_Tensorboard_writer(dir_, use_subfolder=True):

    # No directory provided, so no writer
    if dir_ is None:
        return None

    # Create logger
    else:

        # Create subfolder
        if use_subfolder:
            dt = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            dir_ = os.path.join(dir_, 'Tensorboard_{}'.format(dt))
            os.makedirs(dir_)

        return SummaryWriter(dir_)



def map_1D(X, map_fn):

    # Map is a dictionary
    if isinstance(map_fn, dict):
        return [map_fn[x] for x in X]

    # Assume map is a function
    else:
        return [map_fn(x) for x in X]

def map_2D(X, map_fn):
    return [map_1D(x, map_fn) for x in X]



def loss_reduction(loss, reduction='sum'):

    # Aggregate loss values
    if isinstance(loss, dict):
        loss = torch.stack([v for _, v in loss.items()])
    else:
        raise ValueError("invalid type, need to fill out function")


    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean()
    else:
        raise ValueError("Invalid reduction method:\t{}".format(reduction))

    return loss


def device_check(tensors):
    devices = set([t.device for t in tensors])
    if len(devices) > 1:
        logging.warn("Tensor devices do not match")
        for i, t in enumerate(tensors):
            logging.warn('\t{} - {}'.format(i, t.device))



def batched_select(tensors, indices):


    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    n = len(tensors)

    # Flatten indices for faster lookup
    flat_indices = util.flatten_and_batch_shift_indices( \
                    indices = indices,
                    sequence_length = tensors[0].size(1))

    selected = []
    for t in tensors:
        s = util.batched_index_select(t, indices, flat_indices)
        selected.append(s)

    if n == 1:
        return selected[0]
    else:
        return tuple(selected)
