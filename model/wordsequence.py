# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-01 15:59:26
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep


# class CNN(nn.Module):
#     def __init__(self, input_dim, filters, kernel_size=3, num_layers=4, dropout=0.2):
#         super(CNN, self).__init__()

#         self.word2cnn = nn.Linear(input_dim, filters)

#         self.cnn = nn.Sequential()
#         for i in range(num_layers):
#             pad_size = int((kernel_size - 1) / 2)
#             layer = nn.Conv1d(
#                 in_channels=filters,
#                 out_channels=filters,
#                 kernel_size=kernel_size,
#                 padding=pad_size,
#             )
#             self.cnn.add_module("layer%d" % i, layer)
#             self.cnn.add_module("relu", nn.ReLU())
#             self.cnn.add_module("dropout", nn.Dropout(dropout))
#             self.cnn.add_module("batchnorm", nn.BatchNorm1d(filters))

#     def forward(self, embeddings):
#         # swap seq_len and embedding
#         word_in = torch.tanh(self.word2cnn(embeddings)).transpose(2,1).contiguous()
#         feature_out = self.cnn(word_in).transpose(2,1).contiguous()
#         # .transpose(1, 2).contiguous()
#         return feature_out


class IDCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        filters,
        kernel_size=3,
        num_blocks=4,
        dilations=[1, 2, 1],
        dropout=0.2,
    ):
        super(IDCNN, self).__init__()

        self.word2cnn = nn.Linear(input_dim, filters)

        dcnn = nn.Sequential()
        for i, dilation in enumerate(dilations):
            pad_size = kernel_size // 2 + dilation - 1
            layer = nn.Conv1d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=pad_size,
            )
            dcnn.add_module("layer%d" % i, layer)
            dcnn.add_module("relu", nn.ReLU())
            dcnn.add_module("dropout", nn.Dropout(dropout))
            dcnn.add_module("batchnorm", nn.BatchNorm1d(filters))

        self.idcnn = nn.Sequential()
        for i in range(num_blocks):
            self.idcnn.add_module("block%i" % i, dcnn)
            self.idcnn.add_module("relu", nn.ReLU())
            self.idcnn.add_module("dropout", nn.Dropout(dropout))
            self.idcnn.add_module("batchnorm", nn.BatchNorm1d(filters))

    def forward(self, embeddings):
        # swap seq_len and embedding
        word_in = torch.tanh(self.word2cnn(embeddings)).transpose(2,1).contiguous()
        output = self.idcnn(word_in).transpose(2,1).contiguous()
        return output


class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..."%(data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        # self.batch_size = data.HP_batch_size
        # self.hidden_dim = data.HP_hidden_dim
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim
        if self.use_char:
            self.input_size += data.HP_char_hidden_dim
            if data.char_feature_extractor == "ALL":
                self.input_size += data.HP_char_hidden_dim

        self.feature_num = data.feature_num
        for idx in range(self.feature_num):
            self.input_size += data.feature_emb_dims[idx]
        self.word_feature_extractor = data.word_feature_extractor
        self.hidden_dim = data.HP_hidden_dim
        if self.word_feature_extractor in {"GRU", "LSTM"}:
            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.
            self.droplstm = nn.Dropout(data.HP_dropout)
            self.lstm_layer = data.HP_lstm_layer
            self.bilstm_flag = data.HP_bilstm
            if self.bilstm_flag:
                self.hidden_dim = data.HP_hidden_dim // 2
            else:
                self.hidden_dim = data.HP_hidden_dim            
            if self.word_feature_extractor == "GRU":
                self.lstm = nn.GRU(self.input_size, self.hidden_dim, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
            else:  # self.word_feature_extractor == "LSTM":
                self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "CNN":
            self.cnn_layer = data.HP_cnn_layer
            self.cnn_kernel = data.HP_cnn_kernel
            print("CNN layer: ", self.cnn_layer)
            self.use_idcnn = data.use_idcnn
            if self.use_idcnn:
                self.cnn_list = IDCNN(
                    self.input_size,
                    self.hidden_dim,
                    kernel_size=self.cnn_kernel,
                    num_blocks=4,
                    dilations=[1, 1, 2],
                    dropout=data.HP_dropout,
                )
            else:
                # self.cnn_list = CNN(
                #     self.input_size,
                #     self.hidden_dim,
                #     kernel_size=self.cnn_kernel,
                #     num_layers=self.cnn_layer,
                #     dropout=data.HP_dropout,
                # )                
                input_dim = self.input_size
                filters = self.hidden_dim
                num_layers = self.cnn_layer
                kernel_size = self.cnn_layer
                dropout = data.HP_dropout
                pad_size = int((kernel_size - 1) / 2)

                self.word2cnn = nn.Linear(input_dim, filters)
                self.cnn = nn.Sequential()
                for i in range(num_layers):
                    self.cnn.add_module("layer%d" % i, 
                        nn.Conv1d(
                            in_channels=filters,
                            out_channels=filters,
                            kernel_size=kernel_size,
                            padding=pad_size,
                        )
                    )
                    self.cnn.add_module("relu", nn.ReLU())
                    self.cnn.add_module("dropout", nn.Dropout(dropout))
                    self.cnn.add_module("batchnorm", nn.BatchNorm1d(filters))                


        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)

        if self.gpu:
            self.hidden2tag = self.hidden2tag.cuda()
            if self.word_feature_extractor == "CNN":
                self.word2cnn = self.word2cnn.cuda()
                self.cnn = self.cnn_list.cuda()
            else:
                self.droplstm = self.droplstm.cuda()
                self.lstm = self.lstm.cuda()


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        
        word_represent = self.wordrep(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ## word_embs (batch_size, seq_len, embed_size)
        if self.word_feature_extractor == "CNN":
            # feature_out = self.cnn_list(word_represent)
            word_in = torch.tanh(self.word2cnn(word_represent)).transpose(2,1).contiguous()
            feature_out = self.cnn(word_in).transpose(2,1).contiguous()            
        else:
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            ## lstm_out (seq_len, seq_len, hidden_size)
            feature_out = self.droplstm(lstm_out.transpose(1,0))
        ## feature_out (batch_size, seq_len, hidden_size)
        outputs = self.hidden2tag(feature_out)
        return outputs

    def sentence_representation(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, ), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ## word_embs (batch_size, seq_len, embed_size)
        batch_size = word_inputs.size(0)
        if self.word_feature_extractor == "CNN":
            cnn_feature = self.cnn_list(word_represent).transpose(2,1).contiguous()
            feature_out = F.max_pool1d(cnn_feature, cnn_feature.size(2)).view(batch_size, -1)
        else:
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            ## lstm_out (seq_len, seq_len, hidden_size)
            ## feature_out (batch_size, hidden_size)
            feature_out = hidden[0].transpose(1,0).contiguous().view(batch_size,-1)
            
        feature_list = [feature_out]
        for idx in range(self.feature_num):
            feature_list.append(self.feature_embeddings[idx](feature_inputs[idx]))
        final_feature = torch.cat(feature_list, 1)
        outputs = self.hidden2tag(self.droplstm(final_feature))
        ## outputs: (batch_size, label_alphabet_size)
        return outputs
