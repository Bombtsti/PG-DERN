from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifer(nn.Module):
    def __init__(self, inp_dim,num_class=2, pre_dropout=0.0):
        super(Classifer, self).__init__()
        self.inp_dim = inp_dim
        self.pre_dropout = pre_dropout

        gnn_inp_dim = self.inp_dim
        if self.pre_dropout > 0:
            self.predrop1 = nn.Dropout(p=self.pre_dropout)

        self.fc1 = nn.Sequential(nn.Linear(gnn_inp_dim, 128), nn.LeakyReLU())
        if self.pre_dropout > 0:
            self.predrop2 = nn.Dropout(p=self.pre_dropout)
        self.fc2 = nn.Linear(128, num_class)


    def forward(self, all_emb, q_emb=None):
        node_feat = all_emb
        if self.pre_dropout > 0:
            node_feat = self.predrop1(node_feat)
        if self.pre_dropout > 0:
            node_feat = self.predrop2(node_feat)

        node_feat = self.fc1(node_feat)

        s_feat = node_feat[:, :-1, :]
        q_feat = node_feat[:, -1, :]

        s_logits = self.fc2(s_feat)
        q_logits = self.fc2(q_feat)

        return s_logits, q_logits
