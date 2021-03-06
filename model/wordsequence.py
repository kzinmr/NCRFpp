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

class WordDropout(nn.Module):
    """ copied from flair.nn
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def __init__(self, dropout_rate=0.05, inplace=False):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)


class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..."%(data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim
        if self.use_char:
            self.input_size += data.HP_char_hidden_dim
            if data.char_feature_extractor == "ALL":
                self.input_size += data.HP_char_hidden_dim

        self.feature_num = data.feature_num
        for idx in range(self.feature_num):
            self.input_size += data.feature_emb_dims[idx]
        self.hidden_dim = data.HP_hidden_dim
        self.word_feature_extractor = data.word_feature_extractor
        self.dropout_rate = data.HP_dropout
        self.word_dropout_rate = data.HP_word_dropout
        if self.word_feature_extractor in {"GRU", "LSTM"}:
            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.
            self.droplstm = nn.Dropout(self.dropout_rate)
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
            if self.word_dropout_rate > 0:
                self.word_dropout = WordDropout(self.word_dropout_rate)
            self.word2cnn = nn.Linear(self.input_size, self.hidden_dim)
            self.cnn_layer = data.HP_cnn_layer
            print("CNN layer: ", self.cnn_layer)
            self.use_idcnn = data.use_idcnn
            self.cnn_kernel = data.HP_cnn_kernel
            if self.use_idcnn:
                self.cnn_list = nn.ModuleList()
                self.cnn_drop_list = nn.ModuleList()
                self.cnn_batchnorm_list = nn.ModuleList()
                self.dcnn_drop_list = nn.ModuleList()
                self.dcnn_batchnorm_list = nn.ModuleList()
                self.dilations = [1, 1, 2]
                for idx in range(self.cnn_layer):
                    dcnn = nn.ModuleList()
                    dcnn_drop = nn.ModuleList()
                    dcnn_batchnorm = nn.ModuleList()
                    for i, dilation in enumerate(self.dilations):
                        pad_size = self.cnn_kernel // 2 + dilation - 1
                        dcnn.append(
                            nn.Conv1d(
                                in_channels=self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.cnn_kernel,
                                dilation=dilation,
                                padding=pad_size,
                            )
                        )
                        dcnn_drop.append(nn.Dropout(self.dropout_rate))
                        dcnn_batchnorm.append(nn.BatchNorm1d(self.hidden_dim))
                    self.dcnn_drop_list.append(dcnn_drop)
                    self.dcnn_batchnorm_list.append(dcnn_batchnorm)
                    self.cnn_list.append(dcnn)
                    self.cnn_drop_list.append(nn.Dropout(self.dropout_rate))
                    self.cnn_batchnorm_list.append(nn.BatchNorm1d(self.hidden_dim))

            else:
                self.cnn_list = nn.ModuleList()
                self.cnn_drop_list = nn.ModuleList()
                self.cnn_batchnorm_list = nn.ModuleList()
                pad_size = int((self.cnn_kernel-1)/2)
                for idx in range(self.cnn_layer):
                    self.cnn_list.append(nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=self.cnn_kernel, padding=pad_size))
                    self.cnn_drop_list.append(nn.Dropout(self.dropout_rate))
                    self.cnn_batchnorm_list.append(nn.BatchNorm1d(self.hidden_dim))

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)

        if self.gpu:
            self.hidden2tag = self.hidden2tag.cuda()
            if self.word_feature_extractor == "CNN":
                if self.word_dropout_rate > 0:
                    self.word_dropout = self.word_dropout.cuda()
                self.word2cnn = self.word2cnn.cuda()
                # self.cnn = self.cnn.cuda()
                for idx in range(self.cnn_layer):
                    if self.use_idcnn:
                        for i, dilation in enumerate(self.dilations):
                            self.cnn_list[idx][i] = self.cnn_list[idx][i].cuda()
                            self.dcnn_drop_list[idx][i] = self.dcnn_drop_list[idx][i].cuda()
                            self.dcnn_batchnorm_list[idx][i] = self.dcnn_batchnorm_list[idx][i].cuda()
                    else:
                        self.cnn_list[idx] = self.cnn_list[idx].cuda()
                    self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                    self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
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
        if self.word_dropout_rate > 0.0:
            word_represent = self.word_dropout(word_represent)        
        ## word_embs (batch_size, seq_len, embed_size)
        if self.word_feature_extractor == "CNN":
            word_in = torch.tanh(self.word2cnn(word_represent)).transpose(2,1).contiguous()
            # feature_out = self.cnn(word_in).transpose(2,1).contiguous()

            cnn_feature = word_in
            for idx in range(self.cnn_layer):
                if self.use_idcnn:
                    for i, dilation in enumerate(self.dilations):
                        cnn_feature = F.relu(self.cnn_list[idx][i](cnn_feature))
                        cnn_feature = self.dcnn_drop_list[idx][i](cnn_feature)
                        cnn_feature = self.dcnn_batchnorm_list[idx][i](cnn_feature)
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))

                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)

            feature_out = cnn_feature.transpose(2,1).contiguous()

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
