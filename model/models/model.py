import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gnns import GNN_Encoder, GCN
from .classifer import Classifer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class ContextMLP(nn.Module):
    def __init__(self):
        super(ContextMLP, self).__init__()

    def forward(self, s_emb, q_emb):
        '''
        param s_emb : support data embedding
        param q_emb : query data embedding
        '''
        n_support = s_emb.size(0)
        n_query = q_emb.size(0)

        s_emb_rep = s_emb.unsqueeze(0).repeat(n_query, 1, 1)
        q_emb_rep = q_emb.unsqueeze(1)
        all_emb = torch.cat((s_emb_rep, q_emb_rep), 1)

        return all_emb, None

class DERN(nn.Module):
    def __init__(self, args):
        super(DERN, self).__init__()
        self.gpu_id = args.gpu_id
        self.device = args.device
        self.k_shot = args.n_shot_train
        self.n_way = 2
        self.n_query = args.n_query
        self.n_property = args.n_property
        self.mol_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm = args.enc_batch_norm)
        if args.pretrained:
            model_file = args.pretrained_weight_path
            if args.enc_gnn != 'gin':
                temp = model_file.split('/')
                model_file = '/'.join(temp[:-1]) +'/'+args.enc_gnn +'_'+ temp[-1]
            print('load pretrained model from', model_file)
            self.mol_encoder.from_pretrained(model_file, self.gpu_id)

        self.encode_projection = ContextMLP()

        self.message_gcn = GCN(args.emb_dim,args.emb_dim)
        inp_dim = args.map_dim
        self.adapt_relation = Classifer(inp_dim=inp_dim,num_class=2,pre_dropout=args.rel_dropout2)


    def relation_forward(self, s_emb, q_emb, s_label):
        s_logits, q_logits = self.adapt_relation(s_emb, q_emb)
        return s_logits, q_logits

    # property-guided feature augmentation
    def ftask_aug(self,s_emb, f_tasks_pos):
        '''
        param s_emb : support data embeddings
        param f_tasks_pos : set of memory features
        '''
        pos_emb = torch.chunk(s_emb, 2)[1]
        neg_emb = torch.chunk(s_emb, 2)[0]
        cur_pro_emb = torch.mean(pos_emb, dim=0, keepdim=True)
        pos_list = []
        for i in range(9):
            pos_list.append(torch.mean(f_tasks_pos[i * self.k_shot:i * self.k_shot + self.k_shot], dim=0, keepdim=True))
        all_pro_emb = torch.cat(pos_list, dim=0)
        sim = F.cosine_similarity(all_pro_emb, cur_pro_emb)
        topk_num = torch.topk(sim, self.n_property)[1]
        index = [i.item() for i in topk_num]
        list = []
        for idx in index:
            emb = f_tasks_pos[idx * self.k_shot:idx * self.k_shot + self.k_shot]
            list.append(emb)
        f_embs = torch.cat(list, dim=0)
        fc_embs = torch.cat((pos_emb, f_embs), dim=0)
        x = fc_embs
        x = x / torch.norm(x, dim=-1, keepdim=True)
        similarity = torch.mm(x, x.T)
        new_pos_emb = self.message_gcn(similarity, fc_embs)
        s_positive_emb = torch.chunk(new_pos_emb, self.n_property+1, dim=0)[0]
        s_emb = torch.cat((neg_emb, s_positive_emb), dim=0)
        return s_emb

    # relation graph feature augmentation
    def fea_aug(self,s_emb_map):
        # graph_num = s_emb_map.size()[0]
        # emb_chunk = torch.chunk(s_emb_map,graph_num,dim=0)
        # list = []
        # for i in range(graph_num):
        #     emb = emb_chunk[i].squeeze(0)
        #     x = emb / torch.norm(emb, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模
        #     similarity = torch.mm(x, x.T)  # 矩阵乘法
        #     new_emb = self.message_gcn(similarity,emb)
        #     list.append(new_emb)
        # s_emb_map = torch.stack(list,dim=0)

        q,s,d = s_emb_map.shape
        emb = s_emb_map.reshape((q*s,d))
        x = emb / torch.norm(emb, dim=-1, keepdim=True)
        soft_adj = torch.mm(x, x.T)
        new_emb = self.message_gcn(soft_adj, emb)
        s_emb_map = new_emb.reshape((q, s, d))
        return s_emb_map

    def forward(self, task_id, s_data, q_data,f_tasks_pos, s_label=None):
        '''
        param s_data: support data
        param q_data: query data
        param f_tasks_pos: memory features
        '''

        # extract the embeddings
        s_emb, sub_s_emb, cat_s_emb = self.mol_encoder(s_data)
        q_emb, sub_q_emb, cat_q_emb = self.mol_encoder(q_data)
        pos_emb = torch.chunk(cat_s_emb, 2)[1]
        # property-guided feature augmentation
        if not f_tasks_pos is None:
            s_emb = self.ftask_aug(s_emb,f_tasks_pos)
            sub_s_emb = self.ftask_aug(sub_s_emb,f_tasks_pos)
            cat_s_emb = self.ftask_aug(cat_s_emb,f_tasks_pos)

        # construct relation graph
        s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)
        sub_s_emb_map,sub_q_emb_map = self.encode_projection(sub_s_emb,sub_q_emb)
        cat_s_emb_map, cat_q_emb_map = self.encode_projection(cat_s_emb, cat_q_emb)

        # relation graph feature augmentation
        s_emb_map = self.fea_aug(s_emb_map)
        sub_s_emb_map = self.fea_aug(sub_s_emb_map)
        cat_s_emb_map = self.fea_aug(cat_s_emb_map)

        # predict the results
        s_logits, q_logits = self.relation_forward(s_emb_map, q_emb_map, s_label)
        subs_logits, subq_logits = self.relation_forward(sub_s_emb_map, sub_q_emb_map, s_label)
        cats_logits, catq_logits = self.relation_forward(cat_s_emb_map, cat_q_emb_map, s_label)

        return s_logits, q_logits, subs_logits, subq_logits, cats_logits, catq_logits, pos_emb

    def forward_query_loader(self, taskid,epoch,s_data, q_loader, s_label=None, q_pred_adj=False):
        '''
        param s_data: support data
        param q_loader: query data
        '''

        # extract the embeddings
        s_emb, sub_s_emb, cat_s_emb = self.mol_encoder(s_data)

        y_true_list=[]
        q_logits_list = []
        q_feats_list = []
        for q_data in q_loader:
            q_data = q_data.to(s_emb.device)
            y_true_list.append(q_data.y)

            # extract the embeddings
            q_emb,sub_q_emb,cat_q_emb = self.mol_encoder(q_data)

            # construct relation graph
            cat_s_emb_map, cat_q_emb_map = self.encode_projection(cat_s_emb, cat_q_emb)

            # relation graph feature augmentation
            cat_s_emb_map = self.fea_aug(cat_s_emb_map)

            # predict the results
            cats_logit, catq_logit = self.relation_forward(cat_s_emb_map, cat_q_emb_map, s_label)

            q_logits_list.append(catq_logit)

            q_feat = cat_s_emb_map[:, -1, :]
            q_feats_list.append(q_feat)

        q_logits = torch.cat(q_logits_list, 0)
        y_true = torch.cat(y_true_list, 0)

        return cats_logit, q_logits, y_true