




from transformers import AutoTokenizer, AutoModel, AutoConfig

import torch
import torch.utils.data as data_utils
# from pytorch_transformers import BertModel, BertTokenizer
# from pytorch_transformers import XLNetModel, XLNetTokenizer
from tqdm import tqdm
import numpy as np
import logging

from models.utils import trunc_seqs, pad_sequences
# from models.utils import get_device
# from models.similarity import normalize_embed

CLS = "[CLS]"
SEP = "[SEP]"

# from constants import BERT, XLNET



def get_tokenizer(xfmr_type, xfmr_dir):
    '''
    Get tokenizer, based on model type and directory
    '''

    # Create tokenizer
    if xfmr_type == BERT:
        return BertTokenizer.from_pretrained(xfmr_dir)
    elif xfmr_type == XLNET:
        return XLNetTokenizer.from_pretrained(xfmr_dir)
    else:
        raise ValueError("Invalid xfmr_type: {}".format(xfmr_type))

def get_model(xfmr_type, xfmr_dir):
    '''
    Get tokenizer, based on model type and directory
    '''
    # Create tokenizer
    if xfmr_type == BERT:
        return BertModel.from_pretrained(xfmr_dir)
    elif xfmr_type == XLNET:
        return XLNetModel.from_pretrained(xfmr_dir)
    else:
        raise ValueError("Invalid xfmr_type: {}".format(xfmr_type))


def docs2wordpiece(docs, xfmr_type, xfmr_dir):
    '''
    Convert tokenized documents to word pieces
    '''

    assert isinstance(docs, list)
    assert isinstance(docs[0], list)

    # Get tokenizer
    tokenizer = get_tokenizer(xfmr_type, xfmr_dir)

    # loop on documents
    word_pieces = []
    for doc in docs:

        # Start of doc
        wrd_pc = [CLS]

        # Loop on tokens
        for sent in doc:

            # Text
            text = ' '.join(sent)

            # Convert to word pieces
            wrd_pc.extend(tokenizer.tokenize(text))

            # Sentence boundary
            wrd_pc.append(SEP)

        # Start of sequence padding
        word_pieces.append(wrd_pc)

    # Word piece IDs (vocab indices)
    # Convert tokens to vocab IDs
    word_piece_ids = []
    for wrd_pc in word_pieces:
        word_piece_ids.append(tokenizer.convert_tokens_to_ids(wrd_pc))

    return (word_pieces, word_piece_ids)

def tokens2wordpiece(tokens, tokenizer_path, get_last=True):
    '''
    Get word-word pieces and alignment for sentences
    '''

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Word piece tokens
    word_pieces = []

    # Token map will be an int -> int mapping between the `orig_tokens`
    # index and the `bert_tokens` index.
    token_indices = []

    # Loop on tokens
    for sent in tokens:

        # Start of sequence padding
        tok_idxs = [0]
        wrd_pc = [CLS]

        # Loop on sentence
        for tok in sent:

            # Get indice of first word piece
            if not get_last:
                tok_idxs.append(len(wrd_pc))

            # Convert to word pieces
            wrd_pc.extend(tokenizer.tokenize(tok))

            # Get indice of last word piece
            if get_last:
                tok_idxs.append(len(wrd_pc) - 1)

        # Start of sequence padding
        tok_idxs.append(len(wrd_pc))
        wrd_pc.append(SEP)

        # Append sentence information
        token_indices.append(tok_idxs)
        word_pieces.append(wrd_pc)

    # Word piece IDs (vocab indices)
    # Convert tokens to vocab IDs
    word_piece_ids = []
    for wrd_pc in word_pieces:
        word_piece_ids.append(tokenizer.convert_tokens_to_ids(wrd_pc))


    logging.info('XFMR word pieces generated:')
    for i, (wt, wi, ti) in enumerate(zip(word_pieces, word_piece_ids, token_indices)):
        if i > 5:
            break
        logging.info('Example: {} ------------'.format(i))
        logging.info('word pieces:\t{}'.format(wt))
        logging.info('word pieces ids:\t{}'.format(wi))
        logging.info('token indices:\t{}'.format(ti))


    wp_toks_max_len = max([len(w) for w in word_pieces])
    tok_idx_max = max([max(w) for w in token_indices])
    tok_idx_max_len = max([len(w) for w in token_indices])

    logging.info('')
    logging.info('Max word pieces sent length:\t{}'.format(wp_toks_max_len))
    logging.info('Max word piece index:\t{}'.format(tok_idx_max))
    logging.info('Max token index seq length:\t{}'.format(tok_idx_max_len))


    return (word_pieces, word_piece_ids, token_indices)


def embed_len_check(X, Y):

    assert len(X) == len(Y), '{} vs {}'.format(len(X), len(Y))
    for x, y in zip(X, Y):
        assert len(x) == len(y)-2, '{} vs {}'.format(len(x), len(y)-2)
    return True

#def pad2D(X, max_seq_len):
#
#
#    dtype = type(X[0][0])
#
#    Y = []
#    for x in X:
#
#        # Initialize array of zeros with desired size
#        padded = np.zeros(max_seq_len, dtype=dtype)
#
#        # Insert variable length array into fixed/padded array
#        x = np.array(x, dtype=dtype)
#        r = x.shape[0]
#        padded[:r] = x
#        Y.append(padded)
#
#    # Concatenate along new dimension
#    Y = np.stack(Y, axis=0)
#
#    return Y

def get_attn_mask(X, seq_len, dtype=np.int32):

    # Loop on sequences
    Y = []
    for x in X:
        y = np.zeros(seq_len, dtype=dtype)
        y[0:len(x)] = 1
        x_len = min(len(x), seq_len)
        y_sum = sum(y)
        assert x_len == y_sum, "{} vs {}".format(x_len, y_sum)
        Y.append(y)

    # Concatenate along new dimension
    Y = np.stack(Y, axis=0)

    return Y




def get_embeddings(word_piece_ids, tok_idx, pretrained_path, \
        max_len = None,
        num_workers = 6,
        batch_size = 100,
        device = None):

    tok_idx_fill = -1

    logging.info('='*72)
    logging.info("Generating xfmr embeddings")
    logging.info('='*72)

    # Get device
    if device is None:
        device = get_device()

    # Load pre-trained model (weights)
    logging.info('Loading xfmr...')
    logging.info('pretrained_path:\t{}'.format(pretrained_path))
    model = AutoModel.from_pretrained(pretrained_path)
    logging.info('xfmr loaded')

    # Set model to evaluation mode (as opposed to train mode)
    model.eval()

    # Set model device
    model.to(device)

    # If token indices not provided, use all
    if tok_idx is None:
        logging.info('No token indices provided, so outputting all word pieces')
        tok_idx = [list(range(0, len(sent))) for sent in word_piece_ids]
    else:
        logging.info('Token indices provided, so outputting subset of word pieces')

    # Get maximum length
    if max_len is None:
        max_len = max([len(w) for w in word_piece_ids])

    # Create attention mask, indicating valid token positions
    attn_mask = get_attn_mask(word_piece_ids, max_len)

    # Pad
    word_piece_ids = pad_sequences(word_piece_ids, max_len)
    tok_idx = pad_sequences(tok_idx, max_len, fill=tok_idx_fill)

    # Convert to tensor
    word_piece_ids = torch.tensor(word_piece_ids)
    tok_idx = torch.tensor(tok_idx)
    attn_mask = torch.tensor(attn_mask)

    # Create data set
    dataset = data_utils.TensorDataset(word_piece_ids, attn_mask, tok_idx)

    # Create data loader
    dataloader = data_utils.DataLoader(dataset, \
                batch_size=batch_size,
                shuffle = False,
                num_workers=num_workers)

    # Create progress bar
    pbar = tqdm(total=len(dataloader))

    # Loop on batches
    # Predict hidden states features for each layer
    embeddings = []
    for wp_ids_bat, attn_mask_bat, tok_idx_bat in dataloader:

        # Get embeddings (no gradient)
        with torch.no_grad():
            embed_bat = model( \
                  input_ids = wp_ids_bat.to(device),
                  token_type_ids = None,
                  attention_mask = attn_mask_bat.to(device))[0]

        # Convert to numpy array
        embed_bat = embed_bat.cpu().numpy()
        tok_idx_bat = tok_idx_bat.cpu().numpy()

        # Loop on sequences and extract relevant embeddings
        for embed_seq, tok_idx_seq in zip(embed_bat, tok_idx_bat):

            # Remove trailing zeros
            tok_idx_seq = tok_idx_seq[tok_idx_seq != tok_idx_fill]

            # Get embeddings associated with specific indices
            embed_seq = embed_seq[tok_idx_seq, :]

            embeddings.append(embed_seq)

        # Update progress bar
        pbar.update()

    pbar.close()


    logging.info('XFMR embeddings generated:')
    for i, y in enumerate(embeddings):
        if i > 5:
            break
        logging.info('embed:\t{}'.format(y))

    return embeddings



def get_doc_embed(docs, xfmr_type, xfmr_dir, seq_length, \
        num_workers=6, batch_size=40, normalize=False):


    logging.info('')
    logging.info("GET DOC EMBEDDINGS FROM XFMR")

    '''
    Model setup
    '''
    # Get device
    device = get_device()

    # Load pre-trained model (weights)
    model = get_model(xfmr_type, xfmr_dir)

    # Set model to evaluation mode (as opposed to train mode)
    model.eval()

    # Set model device
    model.to(device)

    '''
    Prepocessing
    '''
    # Get word pieces
    wp_toks, wp_ids = docs2wordpiece( \
                            docs = docs,
                            xfmr_type = xfmr_type,
                            xfmr_dir = xfmr_dir)
    assert len(docs) == len(wp_ids), "{} vs {}".format(len(docs), len(wp_ids))
    logging.info("\tWords mapped to IDs")

    # Truncate long documents
    wp_ids = trunc_seqs(wp_ids, seq_length, bin_size=20)
    logging.info("Sequences truncated")
    assert len(docs) == len(wp_ids), "{} vs {}".format(len(docs), len(wp_ids))

    # Double check token mapping
    for w in wp_toks:
        assert w[0] == CLS

    '''
    Push through xfmr
    '''

    # Pad
    attn_mask = torch.tensor(get_attn_mask(wp_ids, seq_length))
    wp_ids = torch.tensor(pad_sequences(wp_ids, seq_length))

    # Create data set
    dataset = data_utils.TensorDataset(wp_ids, attn_mask)

    # Create data loader
    dataloader = data_utils.DataLoader(dataset, \
                batch_size=batch_size,
                shuffle = False,
                num_workers=num_workers)

    # Create progress bar
    pbar = tqdm(total=len(dataloader))

    # Loop on batches
    # Predict hidden states features for each layer
    embeddings = []
    for wp_ids_bat, attn_mask_bat in dataloader:

        # Get embeddings (no gradient)
        with torch.no_grad():

            # Sequence embedding
            # (batch_size, seq_len, embed_dim)
            seq_embed = model( \
                  input_ids = wp_ids_bat.to(device),
                  token_type_ids = None,
                  attention_mask = attn_mask_bat.to(device))[0]

            # Document embedding
            # (batch_size, embed_dim)
            doc_embed = seq_embed[:,0,:]

            # Normalize
            if normalize:
                doc_embed = normalize_embed(doc_embed)

        # Convert to numpy array
        embeddings.extend(doc_embed.tolist())

        # Update progress bar
        pbar.update()

    pbar.close()
    return embeddings
