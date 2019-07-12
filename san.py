import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from attention import Attention
from ques_emb import QuestionEmbedding

class LSTM_Att(nn.Module):
    def __init__(self, options):
        super(LSTM_Att, self).__init__() 

        self.ques_emb=QuestionEmbedding(options)
        self.image_mlp_act=nn.Linear(options['n_image_feat'], options['n_dim'])
        self.att1=Attention(options)
        self.att1=Attention(options)
        self.combined_mlp_drop_0=nn.Dropout(p=options['drop_ratio'])
        self.combined_mlp_0=nn.Linear(options['n_dim'], options['n_output'])
       

    def forward(self, image_feat, input_idx, ques_len):           
        h_encode = self.ques_emb(input_idx,ques_len)
        image_feat_down = torch.tanh(self.image_mlp_act(image_feat))

        combined_hidden_att1=self.att1(image_feat, h_encode)
        combined_hidden_att2=self.att2(image_feat,combined_hidden_att1)

        combined_hidden=self.combined_mlp_drop_0(combined_hidden_att2)
        combined_hidden=self.combined_mlp_0(combined_hidden)
        
        return combined_hidden