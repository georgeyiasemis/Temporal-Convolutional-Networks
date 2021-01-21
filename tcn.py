import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from pytorch_model_summary import summary
import numpy as np
import math

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
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.transpose = Transpose_Layer(-1, -2)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
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


class AttentionBlock(nn.Module):

    def __init__(self, input_dim, keys_size, values_size):

        """An attention block that takes as an input a tensor of shape
         (N, seq_len, input_dim) where 'N' is the batch size, 'seq_len'
         is the sequence lengtth and 'input_dim' is the dimensions of the
         features or channels and applies a scaled dot product
         attention mechanism.

         The output is a tensor of shape (N, seq_len, input_dim + values_size).

        Parameters
        ----------
        input_dim : int
            Features or channels size.
        keys_size : int
            Dimention of attention keys/queries.
        values_size : type
            Dimension of attention values.

        References:
            https://openreview.net/pdf?id=B1DmUzWAW


        """

        super(AttentionBlock, self).__init__()
        self.key_layer = nn.Linear(input_dim, keys_size)
        self.query_layer = nn.Linear(input_dim, keys_size)
        self.value_layer = nn.Linear(input_dim, values_size)
        self.sqrt_k = np.sqrt(keys_size)

    def forward(self, x):

        keys = self.key_layer(x)
        queries = self.query_layer(x)
        values = self.value_layer(x)

        scores = torch.bmm(queries, keys.transpose(2,1)) / self.sqrt_k
        mask = torch.triu(torch.ones(scores.size()), 1)#.to(device)

        scores.data.masked_fill_(mask, -1e9)
        probs = torch.softmax(scores, dim=1)
        print(probs.shape)
        attention = torch.bmm(probs, values)
        print(x.shape, attention.shape)
        return torch.cat((x, attention), dim=2)


class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_outputs, num_channels, seq_len, kernel_size=2, dropout=0.2, attention=True):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)
        for layer in range(num_levels):
            dilation = 2 ** layer
            in_channels = num_inputs if layer == 0 else num_channels[layer-1]
            out_channels = num_channels[layer]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                     padding=(kernel_size-1) * dilation, dropout=dropout)]
        self.network = nn.Sequential(*layers)

        if attention:
            self.attention = AttentionBlock(num_channels[-1], num_channels[-1], num_channels[-1])
        else:
            self.attention = None

        self.fc = nn.Sequential(weight_norm(nn.Linear(seq_len * num_channels[-1] * 2, num_outputs)),
            nn.ReLU()
            )



    def forward(self, x):
        """
        Parameters
        ----------
        x : tensor
            Input of shape (N, seq_len, num_inputs).

        Returns
        -------
        y : tensor
            Output of shape (N, seq_len, num_channels[-1]).

        """
        y = self.network(x)
        if self.attention is not None:
            y = self.attention(y)
        y = y.reshape(y.shape[0], -1)
        return self.fc(y)

if __name__ == '__main__':

    # model = Chomp1d(3)
    # x = torch.randn(128,  12, 24)
    # print(model(x).shape)

    # model = TemporalBlock(n_inputs=14, n_outputs=1, kernel_size=2, stride=1, dilation=1, padding=1)
    model =  TemporalConvNet(10, 1, (16,128, 1), 12)
    # model = nn.Sequential(AttentionBlock(10,10,10))
#     # model = Transpose_Layer(1,2)
    x = torch.randn(19,  12, 10)
#     # print(x)
#     print(model(x).shape)
#     # x = x.transpose(1,2)
#     # model.training = True
#     # print(model(x)[0].shape,model(x)[1].shape, model.training)
    print(summary(model, x , show_hierarchical=False, show_input=True, show_parent_layers=True))
    print(summary(model, x , show_hierarchical=False, show_input=False, show_parent_layers=True))
# #     m = Conv2d_Block(3, 16, kernel_size=(2,1), padding=(2,1))
# #     print(m(x).shape)
