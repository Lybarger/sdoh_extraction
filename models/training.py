
import torch.nn.functional as F
import torch

import torch
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical
import logging
import math


class SimpleLoss(nn.Module):
    '''
    Weighted loss for multitask learning
    '''

    def __init__(self):
        super(SimpleLoss, self).__init__()

    def __str__(self):
        return str(self.__dict__)

    def forward(self, loss):

        # Total batch loss
        loss_weighted = []
        coefficients = {}
        for k, ls in loss.items():

            coefficients[k] = 1.0

            # Accumulate loss
            loss_weighted.append(ls)

        # Calculate loss total
        loss_total = torch.stack(loss_weighted).sum()

        return (loss_total, coefficients)

class MultiTaskLoss(nn.Module):
    '''
    Weighted loss for multitask learning

     Create penalty for learning objective to balance loss
        propagation for all tasks, based on paper "Multi-task learning
        using uncertainty to weigh losses for scene geometry and
        semantics" by Kendall.
    '''

    def __init__(self, keys_):
        super(MultiTaskLoss, self).__init__()

        self.keys_ = list(keys_)

        # Initialize penalty, so loss coefficient is 1 and
        # penalty = 0
        self.p = math.log(1)/2

        # Create trainable penalty parameters for each task
        self.penalties = {}
        for k in self.keys_:
            setattr(self, self.penalty_name(k), nn.Parameter(torch.FloatTensor([self.p])))
            self.penalties[k] = getattr(self, self.penalty_name(k))

        logging.info('')
        logging.info('MultiTaskLoss - instantiation')
        logging.info('\tkeys:\t{}'.format(self.keys_))
        logging.info('\tp:\t{}'.format(self.p))
        logging.info('')

    def __str__(self):
        return str(self.__dict__)

    def penalty_name(self, k):
        return 'penalty_{}'.format(k)



    def forward(self, loss):

        # Total batch loss
        loss_weighted = []
        coefficients = {}
        for k, ls in loss.items():

            # Loss penalty
            # p = log(sigma)
            p = self.penalties[k]

            # Loss coefficient
            # c = 1/sigma^2 = exp(-2p)
            c = torch.exp(-2*p)
            coefficients[k] = c.item()

            # Accumulate loss
            loss_weighted.append(ls*c + p)

        # Calculate loss total
        loss_total = torch.stack(loss_weighted).sum()

        return (loss_total, coefficients)

def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).


    Copied from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html
    Modified to output clipping coefficient
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return (total_norm, clip_coef)


def get_loss(scores, labels, mask, reduction='sum'):
    '''
    Calculate loss for arguments (e.g. trigger-entity pairs)

    Parameters
    ----------
    scores: (batch_size, trig_num, entity_num, tag_num)
    labels: (batch_size, trig_num, entity_num)
    mask: (batch_size, trig_num, entity_num)

    Returns
    -------
    loss: scalar
    '''

    # Flatten fields
    # (batch_size*trig_num*entity_num,  tag_num)
    tag_num = scores.size(-1)
    scores_flat = scores.view(-1, tag_num)
    # (batch_size*trig_num*entity_num)
    labels_flat = labels.view(-1)
    # (batch_size*trig_num*entity_num)
    mask_flat = mask.view(-1).bool()

    # Loss
    loss = F.cross_entropy(scores_flat[mask_flat],
                           labels_flat[mask_flat],
                           reduction = reduction)
    return loss

def cross_entropy_soft_labels(y_true, y_pred, mask, reduction='sum'):
    '''
    Calculate cross entropy loss for soft labels (label probabilities)
    '''

    # Change data type
    mask = mask.float()

    # Clamp predictions to avoid log(0) and log(1)
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)

    # Calculate cross entropy loss, using mask
    cxe = -((y_true*torch.log(y_pred))*mask).sum(-1)

    # Reduce loss
    cxe = reduce_loss(cxe, reduction)

    return cxe


def reduce_loss(loss, reduction='sum'):

    if isinstance(loss, list):
        loss = torch.stack(loss)

    if reduction == 'sum':
        loss = torch.sum(loss)
    elif reduction == 'mean':
        loss = torch.mean(loss)
    else:
        raise ValueError("Invalid loss reduction:\t{}".format(reduction))

    return loss
#
#def get_entity_loss(scores, labels, mask, reduction='mean'):
#    '''
#    Calculate loss for entity, trigger, or status
#
#    Parameters
#    ----------
#    scores: (batch_size, entity_num, tag_num)
#    labels: (batch_size, entity_num)
#    mask: (batch_size, entity_num)
#
#    Returns
#    -------
#    loss: scalar
#    '''
#
#    # Dimensionality
#    batch_size, entity_num, tag_num = tuple(scores.shape)
#
#    # Flatten fields
#    # (batch_size*entity_num,  tag_num)
#    scores_flat = scores.view(-1, tag_num)
#    # (batch_size*entity_num)
#    labels_flat = labels.view(-1)
#    # (batch_size*entity_num)
#    mask_flat = mask.view(-1).bool()
#
#    # Loss
#    loss = F.cross_entropy(scores_flat[mask_flat],
#                           labels_flat[mask_flat],
#                           reduction = reduction)
#
#    return loss
#
#
#
#def get_arg_loss(scores, labels, mask, reduction='mean'):
#    '''
#    Calculate loss for arguments (e.g. trigger-entity pairs)
#
#    Parameters
#    ----------
#    scores: (batch_size, trig_num, entity_num, tag_num)
#    labels: (batch_size, trig_num, entity_num)
#    mask: (batch_size, trig_num, entity_num)
#
#    Returns
#    -------
#    loss: scalar
#    '''
#
#    # Dimensionality
#    batch_size, trig_num, entity_num, tag_num = tuple(scores.shape)
#
#    # Flatten fields
#    # (batch_size*trig_num*entity_num,  tag_num)
#    scores_flat = scores.view(-1, tag_num)
#    # (batch_size*trig_num*entity_num)
#    labels_flat = labels.view(-1)
#    # (batch_size*trig_num*entity_num)
#    mask_flat = mask.view(-1).byte()
#
#    # Loss
#    loss = F.cross_entropy(scores_flat[mask_flat],
#                           labels_flat[mask_flat],
#                           reduction = reduction)
#
#    return loss
#

    
