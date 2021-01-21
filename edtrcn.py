import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from pytorch_model_summary import summary
import numpy as np
import math
from torch.autograd import Variable

class Transpose_Layer(nn.Module):

    def __init__(self, *dims):
        """
        Trasposes input wrt to the last two dimensions.
        """

        super(Transpose_Layer, self).__init__()
        self.dims = dims

    def forward(self, x):
        """Short summary.

        Parameters
        ----------
        x : tensor
            Input of shape (N, *, M1, M2, *).

        Returns
        -------
        y : tensor
            Input of shape (N, *, M2, M1, *).

        """
        y =  x.transpose(self.dims[0], self.dims[1]).contiguous()

        return y
class Chomp1d(nn.Module):
    '''
    Ensure there is no leakage of information from future to the present.
    '''
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, activation='relu', dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)

        assert activation in ['relu', 'prelu', 'leakyrelu']

        if activation == 'prelu':
            self.relu1 = nn.PReLU()
            self.relu2 = nn.PReLU()
            self.relu = nn.PReLU()

        elif activation == 'leakyrelu':
            self.relu1 = nn.LeakyReLU()
            self.relu2 = nn.LeakyReLU()
            self.relu = nn.LeakyReLU()

        else:
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)

        self.dropout2 = nn.Dropout(dropout)
        self.transpose = Transpose_Layer(-1, -2)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.init_weights()


    def init_weights(self):
        # self.conv1.weight.data.normal_(0, 0.01)
        # self.conv2.weight.data.normal_(0, 0.01)
        # if self.downsample is not None:
            # self.downsample.weight.data.normal_(0, 0.01)
        nn.init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight, gain=np.sqrt(2))

    def forward(self, x):
        """
        Parameters
        ----------
        x : tensor
            Input of shape (N, seq_len, n_inputs).

        Returns
        -------
        y : tensor
            Output of shape (N, seq_len, n_outputs).

        """
        x = self.transpose(x)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.transpose(self.relu(out + res))

class TemporalEncoder(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size, activation='relu', dropout=0.2):
        super(TemporalEncoder, self).__init__()
        layers = []
        num_layers = len(num_channels)
        for layer in range(num_layers):
            dilation = 2 ** layer
            in_channels = num_inputs if layer == 0 else num_channels[layer-1]
            out_channels = num_channels[layer]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                     padding=(kernel_size-1) * dilation, activation=activation, dropout=dropout)]
            layers += [nn.Sequential(
                    Transpose_Layer(-1, -2),
                    nn.MaxPool1d(2),
                    Transpose_Layer(-1, -2))
                    ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out

class TemporalDecoder(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size, activation='relu', dropout=0.2):
        super(TemporalDecoder, self).__init__()
        layers = []
        num_layers = len(num_channels)
        for layer in range(num_layers):
            dilation = 2 ** layer
            in_channels = num_inputs if layer == 0 else num_channels[layer-1]
            out_channels = num_channels[layer]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                     padding=(kernel_size-1) * dilation, activation=activation, dropout=dropout)]
            layers += [nn.Sequential(
                    Transpose_Layer(-1, -2),
                    nn.Upsample(scale_factor=2),
                    Transpose_Layer(-1, -2))
                    ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class SplitLayer(nn.Module):
    '''
    Splits the output of a recurrent module.

    Example:
    --------
    model1 = nn.LSTM(16, 32, batch_first=True)
    model2 = nn.GRU(16, 32, batch_first=True)
    x = torch.randn(20, 12, 16)

    out1, (hidden1, cell) = model1(x)
    out2, hidden2 = model2(x)
    --------- EQUIVALENT TO ---------
    out1 = SplitLayer()(model1(x))
    out2 = SplitLayer()(model2(x))
    '''
    def __init__(self):
        super(SplitLayer, self).__init__()

    def forward(self, x):
        if isinstance(x, tuple) and len(x) >= 2:
            return x[0]
        else:
            return x

class RecurrentLayer(nn.Module):

    def __init__(self, input_dim, hidden_dims=None, mode='gru', dropout=0.2):
        super(RecurrentLayer, self).__init__()
        assert mode in ['gru', 'lstm']
        modules = []
        if hidden_dims == None:
            if mode == 'gru':
                modules.append(nn.GRU(input_dim, input_dim, batch_first=True))
            else:
                modules.append(nn.LSTM(input_dim, input_dim, batch_first=True))
            modules.append(nn.Dropout(dropout))
            modules.append(SplitLayer())
            self.hidden_dims = [input_dim]

        else:
            if isinstance(hidden_dims, int):
                hidden_dims = (hidden_dims, )
            if mode == 'gru':
                modules.append(nn.GRU(input_dim, hidden_dims[0], batch_first=True))
            else:
                modules.append(nn.LSTM(input_dim, hidden_dims[0], batch_first=True))
            modules.append(SplitLayer())
            for i in range(len(hidden_dims)-1):
                if mode == 'gru':
                    modules.append(nn.Dropout(dropout))
                    modules.append(nn.GRU(hidden_dims[i], hidden_dims[i+1], batch_first=True))
                else:
                    modules.append(nn.Dropout(dropout))
                    modules.append(nn.LSTM(hidden_dims[i], hidden_dims[i+1], batch_first=True))
                modules.append(SplitLayer())
            self.hidden_dims = hidden_dims
            self.mode = mode

        self.net = nn.Sequential(*modules)

    def forward(self, x):

        return self.net(x)

class EncoderDecoderCRN(nn.Module):

    def __init__(self, num_inputs, num_outputs, num_channels, recurrent_channels, kernel_size=2, recurrent_mode='gru', activation='relu', dropout=0.2):
        super(EncoderDecoderCRN, self).__init__()
        encoder = TemporalEncoder(num_inputs, num_channels, kernel_size, activation, dropout)

        recurent_hidden_dims =  list(recurrent_channels) + [num_channels[-1]]
        recurrent_layers = RecurrentLayer(num_channels[-1], recurent_hidden_dims, recurrent_mode, dropout=0.2)

        decoder_num_channels = list(num_channels[::-1][1:]) + [num_outputs]
        decoder = TemporalDecoder(num_channels[-1], decoder_num_channels, kernel_size, activation, dropout)

        self.network = nn.Sequential(encoder,  recurrent_layers, decoder)

    def forward(self, x):

        return self.network(x)




if __name__ == '__main__':

    x = torch.randn(30, 16, 25)
    encoder = TemporalEncoder(num_inputs=25, num_channels=(32, 64), kernel_size=3, activation='relu', dropout=0.2)

    # print(summary(encoder, x , show_hierarchical=False, show_input=True, show_parent_layers=True))
    # print(summary(encoder, x , show_hierarchical=False, show_input=False, show_parent_layers=True))
    #
    # y = encoder(x)

    lstm = RecurrentLayer(64, (128, 256, 128, 64), 'lstm')
    # print(summary(lstm, y, show_hierarchical=False, show_input=True, show_parent_layers=True))
    # print(summary(lstm, y, show_hierarchical=False, show_input=False, show_parent_layers=True))
    #
    # y = lstm(y)

    decoder = TemporalDecoder(num_inputs=64, num_channels=(32, 1), kernel_size=3, activation='relu', dropout=0.2)

    # print(summary(decoder, y , show_hierarchical=False, show_input=True, show_parent_layers=True))
    # print(summary(decoder, y , show_hierarchical=False, show_input=False, show_parent_layers=True))

    # model = nn.Sequential(
    #     encoder, lstm, decoder
    # )
    model = EncoderDecoderCRN(25, 1, (32, 64), (128, 256, 128))
    print(summary(model, x, show_hierarchical=False, show_input=True, show_parent_layers=True))
    print(summary(model, x, show_hierarchical=False, show_input=False, show_parent_layers=True))
