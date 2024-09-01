import math

import numpy as np
import torch
import torch.nn as nn


class CNNPredictor(nn.Module):
    def __init__(self, type_num, hidden_size, dropout_rate=0.0):
        super(CNNPredictor, self).__init__()
        self.num_filters = 128
        self.kernel_sizes = [5, 4, 3]
        self.v = nn.Linear(hidden_size * 4, 1)
        self.text_cnn = TextCNN(hidden_size, self.num_filters, self.kernel_sizes)
        self.cnn_linear = nn.Linear(self.num_filters * len(self.kernel_sizes), type_num)
        self.Linear = nn.Linear(hidden_size * 4, hidden_size)
        self.hidden = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, context, mask):
        '''
        :param query: [c, e]
        :param context: [b, t, e]
        :param mask: [b, t], 0 if masked
        :return: [b, e]
        '''

        context_ = context.unsqueeze(1).expand(context.size(0), query.size(0), context.size(1),
                                               context.size(2))  # [b, c, t, e]
        query_ = query.unsqueeze(0).unsqueeze(2).expand_as(context_)  # [b, c, t, e]
        hidden = self.hidden(torch.cat([query_, context_, torch.abs(query_ - context_), query_ * context_], dim=-1))
        tanh_hidden = torch.tanh(hidden)
        scores = self.v(tanh_hidden)  # [b, c, t, 1]
        scores = self.dropout(scores)
        mask = (mask < 1).unsqueeze(1).unsqueeze(3).expand_as(scores)  # [b, c, t, 1]
        scores = scores.masked_fill_(mask, -1e10)
        scores = scores.transpose(-1, -2)  # [b, c, 1, t]
        scores = torch.softmax(scores, dim=-1)  # [b, c, 1, t]
        g = torch.matmul(scores, context_).squeeze(2)  # [b, c, e]
        query = query.unsqueeze(0).expand_as(g)  # [b, c, e]
        tanh_hidden2 = torch.tanh(self.hidden(torch.cat([query, g, torch.abs(query - g), query * g], dim=-1)))
        tanh_hidden2 = self.Linear(tanh_hidden2)
        cnn_hidden = self.text_cnn(tanh_hidden2)
        pred = self.cnn_linear(cnn_hidden)  # [b, c]
        return pred


class Gate(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 3, hidden_size)
        self.dropout = nn.Dropout(config.decoder_dropout)
        # self.linear2 = nn.Linear(hid_size, hid_size)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, y, z):
        '''
        :param x: B, L, K, H
        :param y: B, L, K, H
        :return:
        '''
        o = torch.cat([x, y, z], dim=-1)
        o = self.dropout(o)
        gate = self.linear(o)
        gate = torch.sigmoid(gate)
        # o = gate * x + (1 - gate) * y + 1 * z
        return gate


class TextCNN(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_sizes):
        super(TextCNN, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_channels, num_filters, kernel_size) for kernel_size in kernel_sizes
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        features = []
        x = x.permute(0, 2, 1)
        for conv in self.conv_layers:
            x_conv = conv(x)
            x_conv = self.relu(x_conv)
            x_pool = torch.max(x_conv, dim=2)[0]
            features.append(x_pool)
        features = torch.cat(features, dim=1)
        return features


class QKAttention(nn.Module):
    def __init__(self, hidden_size, heads_num, dropout):
        super(QKAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.Q_layer = nn.Linear(hidden_size, hidden_size)
        self.K_layer = nn.Linear(hidden_size, hidden_size)
        self.V_layer = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)
        self.text_cnn = TextCNN(hidden_size, 128, [5, 4, 3])
        self.cnn_linear = nn.Linear(384, hidden_size)

    def forward(self, query, key, value, mask):
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        query = query.unsqueeze(1)
        query = self.Q_layer(query)
        # key = self.text_cnn(key)
        # key = self.cnn_linear(key).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2)
        key = self.K_layer(key)
        value = self.V_layer(value)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        mask = mask.unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = torch.matmul(probs, value).repeat(1, seq_length, 1)
        # output = torch.matmul(probs, value).squeeze(1)
        output = self.final_linear(output)
        return output


class CNNAttention(nn.Module):
    def __init__(self, hidden_size, heads_num, dropout):
        super(CNNAttention, self).__init__()
        self.num_filters = 128
        self.kernel_sizes = [5, 4, 3]
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.Q_layer = nn.Linear(hidden_size, hidden_size)
        self.K_layer = nn.Linear(hidden_size, hidden_size)
        self.V_layer = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)
        self.text_cnn = TextCNN(hidden_size, self.num_filters, self.kernel_sizes)
        self.cnn_linear = nn.Linear(self.num_filters * len(self.kernel_sizes), hidden_size)

    def forward(self, query, key, value, mask):
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)

        query = query
        query = self.text_cnn(query)
        query = self.cnn_linear(query).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2).squeeze(1)
        # query = self.Q_layer(query)
        key = self.K_layer(key)
        value = self.V_layer(value)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        mask = mask.unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = torch.matmul(probs, value).repeat(1, seq_length, 1)
        output = self.final_linear(output)
        return output

# class QKAttention(nn.Module):
#     def __init__(self, hidden_size, heads_num, dropout):
#         super(QKAttention, self).__init__()
#         self.num_filters = 128
#         self.kernel_sizes = [5, 4, 3]
#         self.hidden_size = hidden_size
#         self.heads_num = heads_num
#         self.per_head_size = hidden_size // heads_num
#
#         self.Q_layer = nn.Linear(hidden_size, hidden_size)
#         self.K_layer = nn.Linear(hidden_size, hidden_size)
#         self.V_layer = nn.Linear(hidden_size, hidden_size)
#
#         self.dropout = nn.Dropout(dropout)
#         self.final_linear = nn.Linear(hidden_size, hidden_size)
#         self.text_cnn = TextCNN(hidden_size, self.num_filters, self.kernel_sizes)
#         self.cnn_linear = nn.Linear(self.num_filters * len(self.kernel_sizes), hidden_size)
#
#     def forward(self, query, key, value, mask):
#         '''
#         query [b,e]
#         key [b,s,e]
#         value [b,s,e]
#         mask [b,s]
#         '''
#         batch_size, seq_length, hidden_size = key.size()
#         heads_num = self.heads_num
#         per_head_size = self.per_head_size
#
#         def shape(x):
#             return x. \
#                 contiguous(). \
#                 view(batch_size, seq_length, heads_num, per_head_size). \
#                 transpose(1, 2)
#
#         def unshape(x):
#             return x.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
#
#         query = query
#         query = self.Q_layer(query).unsqueeze(1).repeat(1, seq_length, 1).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2)
#         # key = self.text_cnn(key)
#         # key = self.cnn_linear(key)
#         key = self.K_layer(key).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2)
#         value = self.V_layer(value).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2)
#
#         scores = torch.matmul(query, key.transpose(-2, -1))
#         scores = scores / math.sqrt(float(per_head_size))
#         mask = mask. \
#             unsqueeze(1). \
#             repeat(1, seq_length, 1). \
#             unsqueeze(1)
#         mask = mask.float()
#         mask = (1.0 - mask) * -10000.0
#         scores = scores + mask
#         probs = nn.Softmax(dim=-1)(scores)
#         probs = self.dropout(probs)
#         output = unshape(torch.matmul(probs, value))
#         output = self.final_linear(output)
#         return output
