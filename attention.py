import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attention(nn.Module): 
    def __init__(self, options):
        super(Attention, self).__init__()
        self.image_att_mlp=nn.Linear(options['n_dim'], options['n_attention'])
        self.sent_att_mlp=nn.Linear(options['n_dim'], options['n_attention'])
        self.droplayer=nn.Dropout(p=options['drop_ratio'])
        self.combined_att_mlp=nn.Linear( options['n_attention'],1)

    def forward(self,img_feat, ques_feat): 
        image_feat_attention=torch.tanh(self.image_att_mlp(img_feat))
        h_encode_attention=torch.tanh(self.sent_att_mlp(ques_feat))
        combined_feat_attention=torch.add(image_feat_attention, \
                        h_encode_attention.view(h_encode_attention.shape[0],1,-1))
        combined_feat_attention=self.droplayer(combined_feat_attention)
        combined_feat_attention=torch.tanh(self.combined_att_mlp(combined_feat_attention))
        prob_attention=F.softmax(combined_feat_attention.view(combined_feat_attention.shape[0],-1),dim=1)
        image_feat_ave = torch.mul(prob_attention.view(prob_attention.shape[0],-1,1) ,img_feat).sum(axis=1)

        combined_hidden = torch.add(image_feat_ave , ques_feat)

        return combined_hidden