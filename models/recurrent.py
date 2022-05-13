




import torch
import torch.nn as nn
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.layer_norm import LayerNorm


class Recurrent(nn.Module):


    def __init__(self, input_size, output_size,
        type_ = 'lstm',
        num_layers = 1,
        bias = True,
        batch_first = True,
        bidirectional = True,
        stateful = False,
        dropout_input = 0.0,
        dropout_rnn = 0.0,
        dropout_output = 0.0,
        layer_norm = True):
        super(Recurrent, self).__init__()


        #device = torch.device("cpu")

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.type_ = str(type_)
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.batch_first = bool(batch_first)
        self.bidirectional = bool(bidirectional)
        self.stateful = bool(stateful)
        self.dropout_input = float(dropout_input)
        self.dropout_rnn = float(dropout_rnn)
        self.dropout_output = float(dropout_output)
        self.layer_norm = bool(layer_norm)


        if self.num_layers == 1:
            assert dropout_rnn == 0


        # Input dropout
        self.drop_layer_input = nn.Dropout(p=dropout_input)

        # Define encoder type
        if type_ == 'lstm':
            encoder = torch.nn.LSTM( \
                        input_size = input_size,
                        hidden_size = output_size,
                        num_layers = num_layers,
                        bias = bias,
                        batch_first = batch_first,
                        dropout = dropout_rnn,
                        bidirectional = bidirectional)
        elif type_ == 'gru':
            encoder = torch.nn.GRU( \
                        input_size = input_size,
                        hidden_size = output_size,
                        num_layers = num_layers,
                        bias = bias,
                        batch_first = batch_first,
                        dropout = dropout_rnn,
                        bidirectional = bidirectional)

        else:
            raise ValueError("incorrect RNN type: {}".format(type_))

        # Create encoder
        self.encoder = PytorchSeq2SeqWrapper( \
                                    module = encoder,
                                    stateful = stateful)

        # Output size
        self.output_size = int(output_size*(1 + int(bidirectional)))

        # Layer normalization
        if self.layer_norm:
            self.normalization = LayerNorm(dimension=self.output_size)

        # Input dropout
        self.drop_layer_output = nn.Dropout(p=dropout_output)



    def forward(self, X, mask, hidden_state=None):


        # Apply dropout to input
        X_drop = self.drop_layer_input(X)

        # Recurrent layer
        h = self.encoder( \
                    inputs = X_drop,
                    mask = mask,
                    hidden_state = hidden_state)

        # Layer normalization
        if self.layer_norm:
            h = self.normalization(h)

        # Apply dropout to output
        h = self.drop_layer_output(h)


        return h
