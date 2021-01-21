import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from pytorch_model_summary import summary
import numpy as np
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

class DotProductAttention(nn.Module):

    def __init__(self, keys_size):
        """A scaled dot product attention block that takes as an input
        keys (K) and queries (Q) of shape (batch_size, *, seq_len, d_k) and
        values (V) of shape (batch_size, *, seq_len, d_v).

        Dimension(s) denotes the number of heads in case of Multi Head Attention
        or can be 1.

        This class when called calculates:

            Attention(Q, K, V) = softmax(score) * V

        where score = Q * K^T / d_k, the scaled dot product of the query and key.

        Parameters
        ----------
        keys_size : int
            Keys/Values dimension (d_k).


        """
        super(DotProductAttention, self).__init__()
        self.sqrt_k = np.sqrt(keys_size)

    def forward(self, queries, keys, values, attention_mask=None):
        """Short summary.

        Parameters
        ----------
        queries : tensor
            Queries tensor of shape (batch_size, *, seq_len, d_k).
        keys : tensor
            Keys tensor of shape (batch_size, *, seq_len, d_k).
        values : tensor
            Values tensor of shape (batch_size, *, seq_len, d_v).
        attention_mask : type, default None
            Mask out all in the input of the softmax function which correspond
            to illegal connections.


        Returns
        -------
        context : tensor
            Attention tensor of shape (batch_size, *, seq_len, d_v).
        attention : tensor
            Scaled dot product tensor of shape (batch_size, *, seq_len, seq_len).

        """
        scores = queries @ keys.transpose(-2,-1) / self.sqrt_k
        if attention_mask is not None:
            scores.data.masked_fill_(attention_mask, -1e9)
        attention = torch.softmax(scores, dim=-1)
        context = attention @ values

        return context, attention

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, input_dim, keys_size=512, values_size=512, num_heads=8, dropout=0.1,
                    use_res_connection=True, mask_attention=True):

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
        values_size : int
            Dimension of attention values.
        num_heads : int
            Number of attention heads
        use_res_connection : bool
            If true a skip connection is used : out = f(context) + x

        References:
            https://openreview.net/pdf?id=B1DmUzWAW
            https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf


        """

        super(MultiHeadAttentionBlock, self).__init__()

        self.key_fc = nn.Linear(input_dim, keys_size * num_heads)
        self.query_fc = nn.Linear(input_dim, keys_size * num_heads)
        self.value_fc = nn.Linear(input_dim, values_size * num_heads)

        self.keys_size = keys_size
        self.values_size = values_size
        self.num_heads = num_heads
        self.use_res_connection = use_res_connection
        self.mask_attention = mask_attention

        self.fc = nn.Linear(num_heads * values_size, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dotattention = DotProductAttention(keys_size)


    def forward(self, x):

        keys = self.key_fc(x).reshape(x.shape[0], self.num_heads, x.shape[1], self.keys_size)
        queries = self.query_fc(x).reshape(x.shape[0], self.num_heads, x.shape[1], self.keys_size)
        values = self.value_fc(x).reshape(x.shape[0], self.num_heads, x.shape[1], self.values_size)

        if self.mask_attention:
            attention_mask = torch.triu(torch.ones(x.shape[0], x.shape[1], x.shape[1]),
                                    diagonal=1).bool().to(device)
            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        else:
            attention_mask = None

        context, attention = self.dotattention(queries, keys, values, attention_mask)
        # (batch_size, seq_len, h * d_v)
        context = context.contiguous().reshape(x.shape[0],
                                            x.shape[1],
                                            self.num_heads * self.values_size)
        out = self.fc(context)
        out = self.dropout(out)

        if self.use_res_connection:
            out = out + x
        else:
            out = out
        return self.layer_norm(out)

class EncoderDecoderTCN(nn.Module):

    def __init__(self, num_inputs, num_outputs, num_channels, kernel_size=2, use_attention=True, activation='relu', dropout=0.2):
        super(EncoderDecoderTCN, self).__init__()
        decoder_num_channels = list(num_channels[::-1][1:]) + [num_outputs]
        encoder = TemporalEncoder(num_inputs, num_channels, kernel_size, activation=activation, dropout=dropout)
        if use_attention:
            attention = MultiHeadAttentionBlock(num_channels[-1], num_channels[-1], num_channels[-1])
        decoder = TemporalDecoder(num_channels[-1], decoder_num_channels, kernel_size, activation=activation, dropout=dropout)

        if use_attention:
            self.network = nn.Sequential(encoder,  attention, decoder)
        else:
            self.network = nn.Sequential(encoder, decoder)
    def forward(self, x):

        return self.network(x)


if __name__ == '__main__':
    x = torch.randn((2,16,15))
    model = EncoderDecoderTCN(15, 1, (12,24,36),2, activation='prelu')
    print(summary(model, x , show_hierarchical=False, show_input=True, show_parent_layers=True))
    print(summary(model, x , show_hierarchical=False, show_input=False, show_parent_layers=True))
