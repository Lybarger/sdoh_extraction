




import torch
#torch.multiprocessing.set_sharing_strategy('file_system')
# torch.backends.cudnn.enabled = False
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
import json
from models.recurrent import Recurrent

from tensorboardX import SummaryWriter

import pandas as pd
import os
import errno
from datetime import datetime
import numpy as np
import logging
from tqdm import tqdm
import joblib
import math
from collections import OrderedDict

from transformers import BertPreTrainedModel
from transformers import BertConfig
from transformers import BertConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig
from models.crf import MultitaskCRF
from models.attention import MultitaskAttention
from models.multitask_dataset import MultitaskDataset, get_label_map
# from models.utils import create_Tensorboard_writer
from models.utils import nested_dict_to_list
# from models.utils import get_device, mem_size
# from models.multitask_dataset import get_label_map
from models.utils import loss_reduction

import config.constants as C





class MultitaskModel(nn.Module):
    '''


    args:
        rnn_type: 'lstm'
        rnn_input_size: The number of expected features in the input x
        rnn_hidden_size: The number of features in the hidden state h
        rnn_num_layers: Number of recurrent layers
        rnn_bias: If False, then the layer does not use bias weights b_ih and b_hh.
        rnn_batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
        rnn_dropout = If non-zero, introduces a Dropout layer on the
                      outputs of each LSTM layer except the last layer,
                      with dropout probability equal to dropout.
        rnn_bidirectional: If True, becomes a bidirectional LSTM.









        https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    '''



    def __init__(self, \

        # Labels
        label_def = None,
        pretrained_path = "emilyalsentzer/Bio_ClinicalBERT",
        tokenizer_path = "emilyalsentzer/Bio_ClinicalBERT",

        # Recurrent layer
        rnn_type = 'lstm',
        rnn_input_size = 768,
        rnn_hidden_size = 100,
        rnn_num_layers = 1,
        rnn_bias = True,
        rnn_input_dropout = 0.0,
        rnn_layer_dropout = 0.0,
        rnn_output_dropout = 0.0,
        rnn_bidirectional = True,
        rnn_stateful = False,
        rnn_layer_norm = True,

        # Attention
        attn_type = 'dot_product',
        attn_size = 100,
        attn_dropout = 0,
        attn_normalize = True,
        attn_activation = None,
        attn_reduction = 'sum',

        # CRF
        crf_constraints = None,
        crf_incl_start_end = True,
        crf_reduction = 'sum',

        # Logging
        log_dir = None,

        # Training
        max_len = 50,
        epochs = 100,
        batch_size = 50,
        num_workers = 6,
        learning_rate = 0.005,
        grad_max_norm = 1,
        overall_reduction = 'sum',

        # Input processing
        pad_start = True,
        pad_end = True,


        ):

        super(MultitaskModel, self).__init__()

        self.input_params = locals()
        del self.input_params["self"]
        del self.input_params["__class__"]


        # Labels
        self.label_def = label_def

        self.pretrained_path = pretrained_path
        self.tokenizer_path = tokenizer_path


        # Recurrent layer
        self.rnn_type = rnn_type
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bias = rnn_bias
        self.rnn_input_dropout = rnn_input_dropout
        self.rnn_layer_dropout = rnn_layer_dropout
        self.rnn_output_dropout = rnn_output_dropout

        self.rnn_bidirectional = rnn_bidirectional
        self.rnn_stateful = rnn_stateful
        self.rnn_layer_norm = rnn_layer_norm
        self.rnn_batch_first = True



        # Attention
        self.attn_type = attn_type
        self.attn_size = attn_size
        self.attn_dropout = attn_dropout
        self.attn_normalize = attn_normalize
        self.attn_activation = attn_activation
        self.attn_reduction = attn_reduction
        self.attn_pred_as_seq = True


        # CRF
        self.crf_constraints = crf_constraints
        self.crf_incl_start_end = crf_incl_start_end
        self.crf_reduction = crf_reduction

        # Logging
        self.log_dir = log_dir

        # Training
        self.max_len = max_len
        self.epochs = epochs

        self.batch_size = batch_size
        self.num_workers = 0 #num_workers
        self.learning_rate = learning_rate
        self.grad_max_norm = grad_max_norm
        self.overall_reduction = overall_reduction

        # Input processing
        self.pad_start = pad_start
        self.pad_end = pad_end


        # Number of tags per label
        _, _, self.num_tags = get_label_map(self.label_def)

        bert_config = AutoConfig.from_pretrained(self.pretrained_path)
        self.bert_size = bert_config.hidden_size

        # Recurrent layer
        self.rnn = Recurrent( \
                        input_size = self.bert_size ,
                        output_size = self.rnn_hidden_size,
                        type_ = self.rnn_type,
                        num_layers = self.rnn_num_layers,
                        bias = self.rnn_bias,
                        batch_first = self.rnn_batch_first,
                        bidirectional = self.rnn_bidirectional,
                        stateful = self.rnn_stateful,
                        dropout_input = self.rnn_input_dropout,
                        dropout_rnn = self.rnn_layer_dropout,
                        dropout_output = self.rnn_output_dropout,
                        layer_norm = self.rnn_layer_norm,
                        )


        # Dictionary repository for output layers
        # NOTE: ModuleDict is an ordered dictionary
        self.out_layers = nn.ModuleDict()
        for name, lab_def in self.label_def.items():

            logging.info("")
            logging.info("-"*72)
            logging.info(name)
            logging.info("-"*72)

            lab_type = lab_def[C.LAB_TYPE]
            event_types = list(self.num_tags[name].keys())
            num_tags = self.num_tags[name]

            # Sentence-level labels
            if lab_type == C.SENT:

                # Probability vectors from previous
                if name == C.TRIGGER:
                    seq_feat_size = None
                elif name in [C.STATUS_TIME, C.STATUS_EMPLOY, C.TYPE_LIVING]:
                    seq_feat_size = self.out_layers[C.TRIGGER].prob_vec_size
                else:
                    seq_feat_size = self.out_layers[C.STATUS_TIME].prob_vec_size + \
                                    self.out_layers[C.STATUS_EMPLOY].prob_vec_size + \
                                    self.out_layers[C.TYPE_LIVING].prob_vec_size

                self.out_layers[name] = MultitaskAttention( \
                        event_types = event_types,
                        num_tags = num_tags,
                        embed_size = self.rnn.output_size,
                        vector_size = self.rnn.output_size,
                        seq_feat_size = seq_feat_size,
                        type_ = self.attn_type,
                        dropout = self.attn_dropout,
                        normalize = self.attn_normalize,
                        activation = self.attn_activation,
                        reduction = self.attn_reduction,
                        pred_as_seq = self.attn_pred_as_seq
                        )

            # cab token-level labels
            elif lab_type == C.SEQ:

                embed_size = self.rnn.output_size + \
                             self.out_layers[C.STATUS_TIME].prob_vec_size + \
                             self.out_layers[C.STATUS_EMPLOY].prob_vec_size + \
                             self.out_layers[C.TYPE_LIVING].prob_vec_size

                self.out_layers[name] = MultitaskCRF( \
                        event_types = event_types,
                        num_tags = num_tags,
                        embed_size = embed_size,
                        constraints = self.crf_constraints,
                        incl_start_end = self.crf_incl_start_end,
                        reduction = self.crf_reduction)

            else:
                raise ValueError("invalid label type:\t{}".format(lab_type))


    def forward(self, X, mask, y=None):
        '''





        Parameters
        ----------
        X: input sequence, already mapped to embeddings

        '''

        '''
        Input layer
        '''
        H = self.rnn(X, mask)


        '''
        Output layers
        '''

        # Predictions
        pred = OrderedDict()

        # Probability vector across all event types
        prob = OrderedDict()

        # Reduced loss across event types (average or sum)
        loss = OrderedDict()

        # Iterate over output layers
        for name, layer in self.out_layers.items():

            # Probability vectors from previous
            if name == C.TRIGGER:
                seq_feats = None
            elif name in [C.STATUS_TIME, C.STATUS_EMPLOY, C.TYPE_LIVING]:
                seq_feats = prob[C.TRIGGER].detach()
            else:
                seq_feats = torch.cat((prob[C.STATUS_TIME], prob[C.STATUS_EMPLOY], prob[C.TYPE_LIVING]),-1).detach()


            pred[name], prob[name], loss[name] = layer( \
                                    X = H,
                                    y = None if y is None else y[name],
                                    mask = mask,
                                    seq_feats = seq_feats)


        return (pred, loss)



    def fit(self, dataset_path, device=None):
        '''
        Train multitask model
        '''

        # Get/set device
        self.to(device)

        # Configure training mode
        self.train()
        logging.info('Fitting model')
        logging.info('self.training = {}'.format(self.training))

        # Print model summary
        self.get_summary()


        # Create data set
        dataset = MultitaskDataset( \
                                dataset_path,
                                label_def = self.label_def,
                                pretrained_path = self.pretrained_path,
                                tokenizer_path = self.tokenizer_path,
                                device = device,
                                max_len = self.max_len,
                                num_workers = self.num_workers,
                                mode = 'fit'
                                )



        # Create data loader
        dataloader = data_utils.DataLoader(dataset, \
                                batch_size = self.batch_size,
                                shuffle = True,
                                num_workers = self.num_workers)

        # Create optimizer
        optimizer = optim.Adam(self.parameters(), \
                                                lr = self.learning_rate)

        # Create logger
        # self.writer = create_Tensorboard_writer( \
        #                             dir_ = self.log_dir,
        #                             use_subfolder = self.log_subfolder)

        # Loop on epochs
        num_epochs_tot = self.epochs
        pbar = tqdm(total=num_epochs_tot)
        j_bat = 0
        for j in range(num_epochs_tot):

            loss_epoch_tot = 0
            loss_epoch_sep = OrderedDict([(k, 0) for k in self.label_def])
            grad_norm_orig = 0
            grad_norm_clip = 0

            # Loop on mini-batches
            for i, (X_, mask_, y_) in  enumerate(dataloader):

                # Reset gradients
                self.zero_grad()

                X_ = X_.to(device)
                mask_ = mask_.to(device)

                for K, V in y_.items():
                    for k, v in V.items():
                        V[k] = v.to(device)


                # Push data through model
                pred, loss = self(X=X_, mask=mask_, y=y_)

                # Total loss across trigger, status, and entity
                loss_bat_tot = loss_reduction(loss, self.overall_reduction)


                # Back probably
                loss_bat_tot.backward()
                grad_norm_orig += clip_grad_norm_(self.parameters(), \
                                                     self.grad_max_norm)
                grad_norm_clip += clip_grad_norm_(self.parameters(), \
                                                           100000000.0)
                optimizer.step()

                # if self.writer is not None:
                #     self.writer.add_scalar('loss_batch', loss_bat_tot, j_bat)
                j_bat += 1

                # Epoch loss
                loss_epoch_tot += loss_bat_tot.item()
                for k in loss_epoch_sep:
                    loss_epoch_sep[k] += loss[k]



            # Average across epochs
            loss_epoch_tot = loss_epoch_tot/i
            for k in loss_epoch_sep:
                loss_epoch_sep[k] += loss_epoch_sep[k]/i
            grad_norm_orig = grad_norm_orig/i
            grad_norm_clip = grad_norm_clip/i

            msg = []
            msg.append('epoch={}'.format(j))
            msg.append('{}={:.1e}'.format('Total', loss_epoch_tot))
            for k, v in loss_epoch_sep.items():
                msg.append('{}={:.1e}'.format(k, v))
            msg = ", ".join(msg)
            pbar.set_description(desc=msg)
            pbar.update()

            # https://github.com/lanpa/tensorboard-pytorch-examples/blob/master/imagenet/main.py
            # if self.writer is not None:
            #     self.writer.add_scalar('loss_epoch_tot', loss_epoch_tot, j)
            #     for k, v in loss_epoch_sep.items():
            #         self.writer.add_scalar('loss_{}'.format(k), v, j)
            #     self.writer.add_scalar('grad_norm_orig', grad_norm_orig, j)
            #     self.writer.add_scalar('grad_norm_clip', grad_norm_clip, j)

        pbar.close()



    def predict(self, dataset_path, device=None, batch_size_pred=200):
        '''
        Train multitask model
        '''



        # Get/set device
        self.to(device)

        # Configure training mode
        self.eval()
        logging.info('Evaluating model')
        logging.info('self.training = {}'.format(self.training))


        # Print model summary
        self.get_summary()

        # Create data set
        dataset = MultitaskDataset( \
                                dataset_path,
                                label_def = self.label_def,
                                pretrained_path = self.pretrained_path,
                                tokenizer_path = self.tokenizer_path,
                                device = device,
                                max_len = self.max_len,
                                num_workers = self.num_workers,
                                mode = 'predict'
                                )


        # Create data loader
        dataloader = data_utils.DataLoader(dataset, \
                                batch_size = batch_size_pred,
                                shuffle = False,
                                num_workers = self.num_workers)

        # Loop on mini-batches
        total = len(dataset)/self.batch_size + 1
        pbar = tqdm(total=total)

        y_pred = []

        for i, (X_, mask_) in enumerate(dataloader):

            X_ = X_.to(device)
            mask_ = mask_.to(device)

            # Push data through model
            pred, loss = self(X=X_, mask=mask_)

            # Convert from pytorch tensor to list
            for K, V in pred.items():
                for k, v in V.items():
                    if isinstance(v, torch.Tensor):
                        V[k] = v.tolist()

            # Convert to list of nested dict
            pred = nested_dict_to_list(pred)

            # Append over batches
            y_pred.extend(pred)


            pbar.update()

        pbar.close()


        # Post process predicitons
        predictions = dataset.decode_(y_pred)

        assert False
        return predictions

    def get_summary(self):

        # Print model summary
        logging.info("\n")
        logging.info("Model summary")
        logging.info(self)

        # Print trainable parameters
        logging.info("\n")
        logging.info("Trainable parameters")
        summary = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                summary.append((name, param.size()))
        df = pd.DataFrame(summary, columns=['Param', 'Dim'])
        logging.info('\n{}'.format(df))

        logging.info("\n")
        num_p = sum(p.numel() for p in self.parameters() \
                                                     if p.requires_grad)
        num_pM = num_p/1e6
        logging.info("Total trainable parameters:\t{:.1f} M".format(num_pM))
        logging.info("\n")


    def save(self, path):
        '''
        Save model
        '''

        if not os.path.exists(path):
            os.makedirs(path)

        # save PyTorch model
        f = os.path.join(path, C.STATE_DICT)
        torch.save(self.state_dict(), f)


        f = os.path.join(path, C.PARAMS_FILE)
        joblib.dump(self.input_params, f)

        return True


def load_multitask_model(path, model_class=MultitaskModel):

    params_file = os.path.join(path, C.PARAMS_FILE)
    state_dict_file = os.path.join(path, C.STATE_DICT)

    assert os.path.exists(params_file)
    assert os.path.exists(state_dict_file)

    logging.info(f"")
    logging.info(f"Loading multitask model...")
    logging.info(f"Model directory: {path}")
    logging.info(f"Parameters file: {params_file}")
    logging.info(f"State dict file: {state_dict_file}")


    input_params = joblib.load(params_file)
    logging.info(f"Parameters loaded:")
    for k, v in input_params.items():
        logging.info(f"\t{k}:\t{v}")

    model = model_class(**input_params)
    logging.info(f"Model instantiated")

    state_dict = torch.load(state_dict_file)
    model.load_state_dict(state_dict)
    logging.info(f"Loaded state dict")
    logging.info("")

    return model
