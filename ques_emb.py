import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class QuestionEmbedding(nn.Module):
    def __init__(self, options):
        super(QuestionEmbedding, self).__init__() 

        self.lstm=nn.LSTM(
                    input_size=options['n_emb'],
                    hidden_size=options['rnn_size'],
                    num_layers=1
                    #batch_first=True    # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)     

                )

        self.w_emb= nn.Embedding(options['n_words']+1, options['n_emb'],padding_idx=0,max_norm=1)


    def forward(self, input_idx, ques_len):           
        input_emb=self.w_emb[input_idx]
        pack_emb = nn.utils.rnn.pack_padded_sequence(input_emb, ques_len, enforce_sorted=False)
        
        encoder_outputs_packed, (h_last, c_last) =self.lstm(pack_emb,None)

        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs_packed)

        h_encode = encoder_outputs
        return h_encode