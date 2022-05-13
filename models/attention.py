

from collections import OrderedDict

import torch
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical
from torch.nn.parameter import Parameter

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.attention import BilinearAttention, DotProductAttention
from allennlp.nn.util import weighted_sum
from allennlp.nn import Activation
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util


from models.utils import loss_reduction
from models.training import get_loss, cross_entropy_soft_labels
# from pytorch_models.span_embedder import span_embed_agg

from tqdm import tqdm
import numpy as np
import logging
from tqdm import tqdm
import joblib
import math
import pandas as pd


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_tags, \
                output_dim = 2,
                reduction = 'sum',
                dropout = 0.0,
                first_tag_is_null = True,
                use_supervised_attn = False,
                path = None,
                span_feat_agg = ['min', 'mean', 'max']):
        super(MultiHeadAttention, self).__init__()

        # Input dimensionality
        self.input_dim = input_dim
        self.num_tags = num_tags
        self.output_dim = output_dim
        self.first_tag_is_null = first_tag_is_null
        self.path_ = path
        self.span_feat_agg = span_feat_agg
        self.span_feat_size = len(span_feat_agg)*num_tags
        self.use_supervised_attn = use_supervised_attn

        # Create attention mechanism
        # Create linear layer without bias
        linear_attn = torch.nn.Linear(input_dim, num_tags, bias=False)
        # Initialize using normal distribution
        torch.nn.init.normal_(linear_attn.weight)
        # Package with AllenNLP
        self.attn = TimeDistributed(linear_attn)

        # Create output layer
        self.lin_output = nn.ModuleDict(OrderedDict())
        for i in range(self.num_tags):
            self.lin_output[str(i)] = torch.nn.Linear(input_dim, output_dim, bias=True)

        # Dropout
        self.drop_layer = nn.Dropout(p=dropout)

        # Cross entropy
        self.reduction = reduction

        # Softmax
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, seq_embed, seq_mask=None, \
                    seq_labels=None, seq_weights=None, verbose=False):

        '''
        Parameters
        ----------
        seq_embed: Input sequence encoding (batch_size, seq_len, hidden_dim)
        seq_mask: Input sequence mask (batch_size, seq_len)
        seq_labels: True labels (batch_size, num_tags)

        Returns
        -------
        probs: Label probabilities (batch_size, num_tags, 2)
        loss: Cross entropy (1)
        '''

        # Get dimensionality
        batch_size, seq_len, hdim = tuple(seq_embed.shape)

        # Calculate raw attention weights
        # shape (batch_size, seq_len, num_tags)
        attn_logits = self.attn(seq_embed)

        # Calculate attention weights as probability
        # shape (batch_size, seq_len, num_tags)
        seq_mask_rep = seq_mask.unsqueeze(-1).repeat(1, 1, self.num_tags)
        attn_weights = util.masked_softmax(attn_logits, seq_mask_rep, dim=1)

        # Transpose last two dimensions
        # shape (batch_size, hidden_dim, seq_len)
        seq_embed = seq_embed.transpose(1, 2)

        # Attended input
        # shape (batch_size, hidden_dim, num_tags)
        attended = torch.bmm(seq_embed, attn_weights)

        # Dropout layer
        # shape (batch_size, hidden_dim, num_tags)
        attended_do = self.drop_layer(attended)

        # Calculate logits, using a separate linear layer for each tag
        logits = []
        for i, x in enumerate(attended_do.split(1, -1)):
            logits_tmp = self.lin_output[str(i)](x.squeeze(-1)).unsqueeze(1)
            logits.append(logits_tmp)
        logits = torch.cat(logits, dim=1)


        # Cross entropy loss
        if seq_labels is not None:
            seq_mask_tmp = torch.ones_like(seq_labels)
            loss = get_loss(logits, seq_labels, seq_mask_tmp, reduction=self.reduction)

            # Incorporate supervised attention  learning
            if self.use_supervised_attn:
                assert seq_weights is not None
                loss_sa = cross_entropy_soft_labels( \
                                        y_true = seq_weights,
                                        y_pred = attn_weights,
                                        mask = seq_mask_rep,
                                        reduction = self.reduction)

                # Aggregate loss
                loss += loss_sa

        else:
            loss = torch.Tensor([0])

        # Get probabilities
        # shape (batch_size, num_tags, output_dim)
        # probs = self.softmax(logits)
        # if self.output_dim == 2:
        #    probs = probs[:, :, 1].squeeze(-1)

        # Get predictions
        # shape (batch_size, num_tags)
        _, pred = logits.max(-1)
        pred = pred.to(logits.device)

        if self.first_tag_is_null:
            pred[:, 0] = (pred[:, 1:].sum(1) == 0).type(pred.type())

        # Attn weights, where weights are zero for pred of 0
        pred_rep = pred.unsqueeze(1).repeat(1, seq_len, 1)
        attn_weights_pos = pred_rep*attn_weights

        if verbose:
            logging.info('MultiHeadAttention - forward')
            logging.info('\tseq_embed:\t{}'.format(seq_embed.shape))
            logging.info('\tseq_mask, original:\t{}'.format(seq_mask.shape))
            logging.info('\tseq_labels, original:\t{}'.format(seq_labels.shape))
            logging.info('\tattn:\t{}'.format(self.attn))
            logging.info('\tseq_mask, repeated:\t{}'.format(seq_mask_rep.shape))
            logging.info('\tAttention weigts:\t{}'.format(attn_weights.shape))
            logging.info('\tseq_embed, transposed:\t{}'.format(seq_embed.shape))
            logging.info('\tAttended input:\t{}'.format(attended.shape))
            logging.info('\tLogits:\t{}'.format(logits.shape))
            #logging.info('\tProbabilities:\t{}'.format(probs.shape))
            logging.info('\tPrediction:\t{}'.format(pred.shape))
            logging.info('\tPrediction, repeated:\t{}'.format(pred_rep.shape))
            logging.info('\tAttention weigts, positive pred:\t{}'.format(attn_weights_pos.shape))

        if (self.path_ is not None) and verbose:
            # shape (batch_size, seq_len, num_tags)
            alpha_ex = []

            batch_size, seq_len, num_tags = tuple(attn_weights_pos.shape)

            for i_seq in range(batch_size):
                for i_tag in range(num_tags):

                    if self.use_supervised_attn:
                        alphas = seq_weights[i_seq,:,i_tag].tolist()
                        x = [i_seq, 'true', i_tag] + alphas
                        alpha_ex.append(tuple(x))

                    alphas = attn_weights_pos[i_seq,:,i_tag].tolist()
                    x = [i_seq, 'pred', i_tag] + alphas
                    alpha_ex.append(tuple(x))
            columns = ['i_seq', 'type', 'i_tag'] + ['x{}'.format(i) for i, _ in enumerate(alphas)]
            df = pd.DataFrame(alpha_ex, columns=columns)
            df.to_csv(self.path_)




        #return (pred, probs, attn_weights_pos, loss)

        return (pred, attn_weights_pos, loss)


    def span_feat(self, seq_embed, seq_mask, seq_labels, span_indices, \
                       seq_weights=None, span_embed=None, verbose=False):
        '''
        Create span embedding from attention weights

        Parameters
        ----------
        span_embed: (batch_size, num_spans, embed_dim)
        '''

        if verbose:
            logging.info('')
            logging.info('AttentionBasedSpan - embed')

        # Run multi head self attention
        # pred (batch_size, num_tags)
        # alphas (batch_size, seq_len, num_tags)
        pred, alphas, loss = self(seq_embed, seq_mask, seq_labels, \
                        seq_weights=seq_weights, verbose=verbose)

        # Aggregate attention weights to create span-level scores
        # (batch_size, num_spans, num_tags)
        span_scores = span_embed_agg(alphas, span_indices,  \
                                            agg = self.span_feat_agg,
                                            verbose = verbose)

        # Concatenate scores with embedding, if embedding provided
        if span_embed is not None:
            # (batch_size, num_spans, embed_dim + num_tags)
            span_scores = torch.cat((span_embed, span_scores), -1)
            if verbose:
                logging.info('\tspan_embed:\t{}'.format(span_embed.shape))

        if verbose:
            logging.info('\tspan_feat_agg:\t{}'.format(self.span_feat_agg))
            logging.info('\tnum_tags:\t{}'.format(self.num_tags))
            logging.info('\tspan_feat_size:\t{}'.format(self.span_feat_size))
            logging.info('\tspan_scores:\t{}'.format(span_scores.shape))

        return (span_scores, alphas, loss)

    def span_logits(self, seq_embed, seq_mask, seq_labels, span_indices, span_labels,
                                    seq_weights=None, verbose=False):

        if verbose:
            logging.info('')
            logging.info('AttentionBasedSpan - pred')

        # Run multi head self attention
        # pred (batch_size, num_tags)
        # alphas (batch_size, seq_len, num_tags)
        pred, alphas, loss = self(seq_embed, seq_mask, seq_labels, \
                        seq_weights=seq_weights, verbose=verbose)

        accuracy = (pred[:,1:] == seq_labels[:,1:]).sum().float()/(seq_labels[:,1:] == seq_labels[:,1:]).sum().float()
        #accuracy = (pred == seq_labels).sum().float()/(seq_labels == seq_labels).sum().float()
        print('')
        print('Accuracy:\t{:.3f}'.format(accuracy), 'Loss:\t{:.3f}'.format(loss.item()))
        print('')

        # Aggregate attention weights to create span-level scores
        # (batch_size, num_spans, num_tags)
        span_scores = span_embed_agg(alphas, span_indices, agg='mean', \
                                                       verbose=verbose)

        # Get indices of spans with max values
        # (batch_size, num_tags)
        batch_size, num_spans, _ = tuple(span_indices.shape)
        _, max_score_idx = span_scores.max(1)

        # Create "fake" logits
        # (batch_size, num_tags, num_spans)
        logits = F.one_hot(max_score_idx, num_classes=num_spans).type(seq_embed.type())
        # (batch_size, num_spans, num_tags)
        logits = torch.transpose(logits, 1, 2).contiguous()

        # Only keep non-zero for postive predictions
        # (batch_size, num_spans, num_tags)
        pred_rep = pred.unsqueeze(1).repeat(1, num_spans, 1)
        logits = logits*pred_rep

        # Set 0 (null) index
        logits[:, :, 0] = (logits[:,:,1:].sum(-1) == 0).type(logits.type())

        if verbose:
            logging.info('\tseq_embed:\t{}, {}'.format(seq_embed.shape, seq_embed.device))
            logging.info('\tseq_mask:\t{}, {}'.format(seq_mask.shape, seq_mask.device))
            logging.info('\tlogits:\t{}, {}'.format(logits.shape, logits.device))
            logging.info('\tloss:\t{}, {}'.format(loss.shape, loss.device))
        return (logits, alphas, loss)



class Attention(nn.Module):
    '''
    Single-task attention
    '''

    def __init__(self, num_tags, embed_size, vector_size, \
        seq_feat_size = None,
        type_ = 'dot_product',
        dropout = 0.0,
        normalize = True,
        activation = 'linear',
        reduction = 'sum',
        include_cnn = False,
        num_filters = 10,
        ngram_filter_sizes = (2,3)):
        super(Attention, self).__init__()


        self.num_tags = num_tags
        self.embed_size = embed_size
        self.vector_size = vector_size
        self.seq_feat_size = seq_feat_size
        self.type_ = type_
        self.dropout = dropout
        self.normalize = normalize
        self.activation = activation
        self.reduction = reduction
        self.include_cnn = include_cnn
        self.num_filters = num_filters
        self.ngram_filter_sizes = ngram_filter_sizes


        # Dot product attention
        if type_ == 'dot_product':
            self.encoder = DotProductAttention(normalize = normalize)
            logging.info('Attention - Overriding vector size for dot product, setting to {}'.format(embed_size))
            logging.info('Attention - Specified activation not used in dot-product attention: {}'.format(self.activation))
            self.vector_size = embed_size
        # BiLinear Attention
        elif type_ == 'bilinear':
            self.encoder = BilinearAttention( \
                    vector_dim = self.vector_size,
                    matrix_dim = embed_size,
                    activation = Activation.by_name(self.activation)(),
                    normalize = normalize)
        else:
            raise ValueError("incorrect type: {}".format(type_))


        # Event-specific attention vector
        # (embed_size)
        self.vector = Parameter(torch.Tensor(self.vector_size))
        torch.nn.init.normal_(self.vector)

        # Optional CNN
        if self.include_cnn:
            self.cnn = CnnEncoder( \
                         embedding_dim = self.embed_size,
                         num_filters = self.num_filters,
                         ngram_filter_sizes = self.ngram_filter_sizes)
            cnn_out_size = self.cnn.get_output_dim()

        # Dropout
        self.drop_layer = nn.Dropout(p=self.dropout)

        # Event-specific output layer (logits to probability)
        inner_dim = embed_size
        if seq_feat_size is not None:
            inner_dim += seq_feat_size
        if self.include_cnn:
            inner_dim += cnn_out_size

        self.out_layer = nn.Linear(inner_dim, num_tags)

        # Softmax
        self.softmax = torch.nn.Softmax(dim=1)

        logging.info('Attention')
        logging.info('\tnum_tags: {}'.format(num_tags))
        logging.info('\tvector_size: {}'.format(self.vector_size))
        logging.info('\tembed_size: {}'.format(embed_size))
        logging.info('\tvector.size(): {}'.format(self.vector.size()))
        logging.info('\tactivation: {}'.format(self.activation))
        logging.info('\tout_layer: {}'.format(self.out_layer))

        # Cross entropy
        self.loss = nn.CrossEntropyLoss(reduction=reduction)


    def forward(self, X, y=None, mask=None, seq_feats=None):
        '''
        Generate predictions


        Parameters
        ----------
        X: input with shape (batch_size, max_seq_len, embed_size)
        mask: input with shape (batch_size, max_seq_len)

        '''

        # Batch size
        batch_size = batch_size = len(mask)

        # Batch vector (repeat across first dimension)
        vector = self.vector.unsqueeze(0).repeat(batch_size, 1)

        # Attention weights
        # shape: (batch_size, max_seq_len)
        alphas = self.encoder( \
                                    vector = vector,
                                    matrix = X,
                                    matrix_mask = mask)

        # Attended input
        # shape: (batch_size, encoder_output_dim)
        input_ = weighted_sum(X, alphas)

        # CNN
        if self.include_cnn:
            cnn_feat = self.cnn(tokens=X, mask=mask)
            input_ = torch.cat((input_, cnn_feat), 1)

        # Append sequence features
        if seq_feats is not None:
            input_ = torch.cat((input_, seq_feats), 1)

        # Dropout layer
        input_ = self.drop_layer(input_)

        # Label socres
        scores = self.out_layer(input_)

        # Predictions
        # (batch_size)
        pred = argmax(scores)

        # Label probabilities
        # (batch_size, num_tags)
        prob = self.softmax(scores)

        # Entropy
        H = Categorical(probs=prob).entropy()

        # Cross entropy loss
        if y is not None:
            loss = self.loss(scores, y)
        else:
            loss = torch.Tensor([0])

        return (alphas, pred, prob, H, loss)




class MultitaskAttention(nn.Module):
    '''
    Multitask attention
    '''
    def __init__(self, event_types, num_tags, embed_size, vector_size, \
        seq_feat_size = None,
        type_ = 'dot_product',
        dropout = 0.0,
        normalize = True,
        activation = 'linear',
        reduction = 'sum',
        pred_as_seq = False,
        include_cnn = False,
        num_filters = 10,
        ngram_filter_sizes = (2,3)):
        super(MultitaskAttention, self).__init__()

        self.event_types = event_types
        self.num_tags = num_tags
        self.embed_size = embed_size
        self.vector_size = vector_size
        self.seq_feat_size = seq_feat_size
        self.type_ = type_
        self.dropout = dropout
        self.normalize = normalize
        self.activation = activation
        self.reduction = reduction
        self.pred_as_seq = pred_as_seq
        self.include_cnn = include_cnn
        self.num_filters = num_filters
        self.ngram_filter_sizes = ngram_filter_sizes

        logging.info('')
        logging.info("Multitask attention")

        # Loop on event_types
        self.encoder = nn.ModuleDict(OrderedDict())
        for t in event_types:

            logging.info('')
            logging.info(t)

            self.encoder[t] = Attention( \
                                num_tags = self.num_tags[t],
                                embed_size = self.embed_size,
                                vector_size = self.vector_size,
                                seq_feat_size = self.seq_feat_size,
                                type_ = self.type_,
                                dropout = self.dropout,
                                normalize = self.normalize,
                                activation = self.activation,
                                reduction = self.reduction,
                                include_cnn = self.include_cnn,
                                num_filters = self.num_filters,
                                ngram_filter_sizes = self.ngram_filter_sizes)

        # Get probability vector output size
        self.prob_vec_size = sum([n for _, n in self.num_tags.items()])

    def forward(self, X, y=None, mask=None, seq_feats=None):
        '''
        Generate predictions
        '''

        # Initialize output dictionaries
        alphas = OrderedDict()  # Attention weights
        pred = OrderedDict()    # Predictions
        prob = OrderedDict()    # Probabilities
        H = OrderedDict()       # Entropy
        loss = OrderedDict()    # Loss

        # Loop on event_types
        for t in self.event_types:
            alphas[t], pred[t], prob[t], H[t], loss[t] = \
                                    self.encoder[t]( \
                                        X = X,
                                        y = None if y is None else y[t],
                                        mask = mask,
                                        seq_feats = seq_feats)

        # Concatenate probabilities across  event types
        prob_vec = torch.cat([t for _, t in prob.items()], dim=-1)


        # Aggregate loss values
        loss = loss_reduction(loss, self.reduction)

        # Provide output as sequence
        if self.pred_as_seq:
            pred_seq = OrderedDict()
            for t in self.event_types:

                # Get indices of maximum
                _, idx = alphas[t].max(1)
                idx.to(alphas[t].device)

                # Initialize batched sequence as zeros
                s = torch.zeros_like(alphas[t], dtype=pred[t].dtype)

                # Insert predicted values at location of maximum
                s.scatter_(1, idx.view(-1, 1), pred[t].view(-1, 1))
                pred_seq[t] = s

            return (pred_seq, prob_vec, loss)

        # Provide output with alphas and predictions separate
        else:
            return (alphas, prob, pred, prob_vec, loss)
