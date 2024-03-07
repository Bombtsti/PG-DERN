import json
import random
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
from torchmetrics.functional import auroc

import task_generator
from chem_lib.datasets import MoleculeDataset
from encoder import GNN_Encoder, GCN1, GCN2
from parser import get_args
warnings.filterwarnings("ignore")

# Hyper Parameters
EPISODE = 5000
TEST_EPISODE = 600
LEARNING_RATE = 0.001
GPU = 0

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv1d(1,32,kernel_size=5,padding=1),
                        nn.BatchNorm1d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(5))
        self.layer2 = nn.Sequential(
                        nn.Conv1d(32,32,kernel_size=5,padding=1),
                        nn.BatchNorm1d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(5))
        self.fc0 = nn.Linear(736, 500)
        self.fc1 = nn.Linear(500,200)
        self.fc2 = nn.Linear(200,1)

    def forward(self,x):
        x = torch.reshape(x,(-1,1,x.shape[1]))
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc0(out))
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out



def weights_init(m):
    classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #     m.weight.data.normal_(0, math.sqrt(2. / n))
    #     if m.bias is not None:
    #         m.bias.data.zero_()
    if classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

class Logger(object):
  def __init__(self, filename="Default.log"):
    self.terminal = sys.stdout
    self.log = open(filename, "a")
  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)
  def flush(self):
    pass

# def fea_aug(s_emb,q_emb,message_gcn):
#     n_support = s_emb.size(0)
#     n_query = q_emb.size(0)
#
#     s_emb_rep = s_emb.unsqueeze(0).repeat(n_query, 1, 1)
#     q_emb_rep = q_emb.unsqueeze(1)
#     s_emb_map = torch.cat((s_emb_rep, q_emb_rep), 1)
#
#     graph_num = s_emb_map.size()[0]
#     emb_chunk = torch.chunk(s_emb_map,graph_num,dim=0)
#     list = []
#     for i in range(graph_num):
#         emb = emb_chunk[i].squeeze(0)
#         x = emb / torch.norm(emb, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模
#         similarity = torch.mm(x, x.T)  # 矩阵乘法
#         new_emb = message_gcn(similarity,emb)
#         list.append(new_emb)
#     s_emb_map = torch.stack(list,dim=0)
#
#     # q,s,d = s_emb_map.shape
#     # emb = s_emb_map.reshape((q*s,d))
#     # x = emb / torch.norm(emb, dim=-1, keepdim=True)
#     # adj = torch.mm(x, x.T)
#     # new_emb = message_gcn(adj, emb)
#     # s_emb_map = new_emb.reshape((q, s, d))
#
#     s_feat = s_emb_map[:, :-1, :]
#     s_feat = s_feat[0, :, :].squeeze(0)
#     q_feat = s_emb_map[:, -1, :].squeeze(1)
#     return s_feat,q_feat

def ftask_aug(s_emb, f_tasks_pos,message_gcn):
    pos_emb = torch.chunk(s_emb, 2)[1]
    neg_emb = torch.chunk(s_emb, 2)[0]
    cur_pro_emb = torch.mean(pos_emb, dim=0, keepdim=True)
    pos_list = []
    for i in range(9):
        pos_list.append(torch.mean(f_tasks_pos[i * 10:i * 10 + 10], dim=0, keepdim=True))
    all_pro_emb = torch.cat(pos_list, dim=0)
    sim = F.cosine_similarity(all_pro_emb, cur_pro_emb)
    topk_num = torch.topk(sim, 3)[1]
    index = [i.item() for i in topk_num]
    list = []
    for idx in index:
        emb = f_tasks_pos[idx * 10:idx * 10 + 10]
        list.append(emb)
    f_embs = torch.cat(list, dim=0)
    fc_embs = torch.cat((pos_emb, f_embs), dim=0)
    x = fc_embs
    x = x / torch.norm(x, dim=-1, keepdim=True)
    similarity = torch.mm(x, x.T)
    new_pos_emb = message_gcn(similarity, fc_embs)
    s_positive_emb = torch.chunk(new_pos_emb, 3+1, dim=0)[0]
    s_emb = torch.cat((neg_emb, s_positive_emb), dim=0)
    return s_emb

def main():
    sys.stdout = Logger('resultrn.txt')
    args = get_args('.')
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    # metatrain_folders,metatest_folders = tg.mini_imagenet_folders()
    preload_train_data = {}
    if args.preload_train_data:
        print('preload train data')
        for task in args.train_tasks:
            dataset = MoleculeDataset(args.data_dir + args.dataset + "/new/" + str(task + 1),
                                      dataset=args.dataset)
            preload_train_data[task] = dataset
    preload_test_data = {}
    if args.preload_test_data:
        print('preload_test_data')
        for task in args.test_tasks:
            dataset = MoleculeDataset(args.data_dir + args.dataset + "/new/" + str(task + 1),
                                      dataset=args.dataset)
            preload_test_data[task] = dataset

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = GNN_Encoder(num_layer=5, emb_dim=300, JK='last',
                                       drop_ratio=0.5, graph_pooling='mean', gnn_type='gin',
                                       batch_norm = 1)
    relation_network = RelationNetwork(300,128)
    message_gcn = GCN2(300,300)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)
    message_gcn.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)
    message_gcn.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)
    message_gcn_optim = torch.optim.Adam(message_gcn.parameters(),lr=LEARNING_RATE)
    message_gcn_scheduler = StepLR(message_gcn_optim,step_size=100000,gamma=0.5)

    if os.path.exists(str("./models/rn10/feature_encoder_" + str(2) +"way_" + str(10) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/feature_encoder_" + str(2) +"way_" + str(10) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/rn/10relation_network_"+ str(2) +"way_" + str(10) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/relation_network_"+ str(2) +"way_" + str(10) +"shot.pkl")))
        print("load relation network success")
    if os.path.exists(str("./models/rn10/message_gcn_"+ str(2) +"way_" + str(10) +"shot.pkl")):
        message_gcn.load_state_dict(torch.load(str("./models/message_gcn_"+ str(2) +"way_" + str(10) +"shot.pkl")))
        print("load message_gcn success")

    # Step 3: build graph
    print("Training...")

    last_scores = 0.0

    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        message_gcn_scheduler.step(episode)

        # init dataset
        task_id_list = list(range(len(args.train_tasks)))
        if args.batch_task > 0:
            batch_task = min(args.batch_task, len(task_id_list))
            task_id_list = random.sample(task_id_list, batch_task)

        total_loss = torch.zeros(1).cuda(GPU)

        start = time.time()

        all_task_pos_embs = []

        for task in task_id_list:
            db = task_generator.MetaTask(task,preload_train_data,'train',2,10,16)

            support_dataloader = db.support_loader[task]
            suppert_labels = db.support_labels[task]
            query_dataloader = db.query_loader[task]
            query_labels = db.query_labels[task]


            node_s_emb,sub_s_emb,s_emb = feature_encoder(support_dataloader)
            node_q_emb, sub_q_emb, q_emb = feature_encoder(query_dataloader)

            pos_emb = torch.chunk(s_emb, 2)[1]
            all_task_pos_embs.append(pos_emb)

            # relation graph augmentation
            node_s_emb,node_q_emb = message_gcn(node_s_emb,node_q_emb)
            sub_s_emb,sub_q_emb = message_gcn(sub_s_emb,sub_q_emb)
            s_emb,q_emb = message_gcn(s_emb,q_emb)

            node_s_emb =node_s_emb.view(2,10,300)
            sub_s_emb = sub_s_emb.view(2,10,300)
            s_emb = s_emb.view(2,10,300)

            node_s_emb = torch.sum(node_s_emb, 1).squeeze(1)
            sub_s_emb = torch.sum(sub_s_emb, 1).squeeze(1)
            s_emb = torch.sum(s_emb, 1).squeeze(1)


            # calculate relations
            # each batch sample link to every samples to calculate relations
            # to form a 100x128 matrix for relation network
            support_features_ext = node_s_emb.unsqueeze(0).repeat(16,1,1)
            query_features_ext = node_q_emb.unsqueeze(0).repeat(2,1,1)
            query_features_ext = torch.transpose(query_features_ext,0,1)
            relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,600)
            relations1 = relation_network(relation_pairs).view(-1,2)

            support_features_ext = sub_s_emb.unsqueeze(0).repeat(16,1,1)
            query_features_ext = sub_q_emb.unsqueeze(0).repeat(2,1,1)
            query_features_ext = torch.transpose(query_features_ext,0,1)
            relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,600)
            relations2 = relation_network(relation_pairs).view(-1,2)

            support_features_ext = s_emb.unsqueeze(0).repeat(16,1,1)
            query_features_ext = q_emb.unsqueeze(0).repeat(2,1,1)
            query_features_ext = torch.transpose(query_features_ext,0,1)
            relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,600)
            relations3 = relation_network(relation_pairs).view(-1,2)

            mse = nn.MSELoss().cuda(GPU)
            l1loss = nn.L1Loss().cuda(GPU)
            one_hot_labels = torch.zeros(16, 2).cuda(GPU).scatter_(1, query_labels.view(-1,1), 1)
            loss = mse(relations3,one_hot_labels)
            loss1 = l1loss(relations1,relations2)
            loss = loss+0.01*loss1
            total_loss+=loss
            # training
        end = time.time()
        if episode%10 == 0:
            print("episode:",episode,"train_loss:",total_loss,"use:",str(end-start))

        feature_encoder.zero_grad()
        relation_network.zero_grad()
        message_gcn.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(message_gcn.parameters(), 0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()
        message_gcn_optim.step()

        if episode%100 == 0:
            all_task_pos_embs = torch.cat(all_task_pos_embs,dim=0)
            # test
            print("Testing...")
            start1 = time.time()
            auc_scores = []

            task_id_list = args.test_tasks
            ts = {}
            for task in task_id_list:
                db = task_generator.MetaTask(task, preload_test_data,'test', 2, 10, 16)

                support_dataloader = db.support_loader[task]
                suppert_labels = db.support_labels[task]
                query_dataloader = db.query_loader[task]
                query_labels = db.query_labels[task]

                node_s_emb, sub_s_emb, s_emb = feature_encoder(support_dataloader)
                # feature augment
                # s_emb = ftask_aug(s_emb,all_task_pos_embs,message_gcn)

                predict_labels = []
                y_true_list = []
                for q_loader in query_dataloader:
                    q_loader = q_loader.cuda(GPU)
                    y_true_list.append(q_loader.y)
                    node_q_emb, sub_q_emb, q_emb = feature_encoder(q_loader)

                    # relation graph augmentation
                    cache_s_emb,q_emb = message_gcn(s_emb,q_emb)

                    cache_s_emb = cache_s_emb.view(2, 10, 300)
                    cache_s_emb = torch.sum(cache_s_emb, 1).squeeze(1)

                    support_features_ext = cache_s_emb.unsqueeze(0).repeat(q_emb.size(0), 1, 1)
                    query_features_ext = q_emb.unsqueeze(0).repeat(2, 1, 1)
                    query_features_ext = torch.transpose(query_features_ext, 0, 1)
                    relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1, 600)
                    relations = relation_network(relation_pairs).view(-1, 2)

                    predict = F.softmax(relations, dim=-1).detach()[:, 1]
                    predict_labels.append(predict)

                y_true = torch.cat(y_true_list, 0)
                prediction = torch.cat(predict_labels, 0)

                ts[task] = {'ys': prediction, 'yt': y_true}


                auc = auroc(prediction, y_true, pos_label=1).item()
                auc_scores.append(auc)

            avg_scores = np.mean(auc_scores)
            end1 = time.time()
            print("avg_scores:",avg_scores,"t1:",auc_scores[0],'t2:',auc_scores[1],"t3:",auc_scores[2],"use:",str(end1 - start1))

            if avg_scores > last_scores:
                # save networks
                torch.save(feature_encoder.state_dict(),str("./models/rn10/feature_encoder_" + str(2) +"way_" + str(10) +"shot.pkl"))
                torch.save(relation_network.state_dict(),str("./models/rn10/relation_network_"+ str(2) +"way_" + str(10) +"shot.pkl"))
                torch.save(message_gcn.state_dict(),str("./models/rn10/message_gcn_" + str(2) + "way_" + str(10) + "shot.pkl"))
                print("save networks for episode:",episode)
                # print(ts)
                # with open('ts.json', 'w') as json_file:
                #     for key, tensors in ts.items():
                #         json.dump({key: tensor.tolist() for key, tensor in tensors.items()}, json_file)

                last_scores = avg_scores


if __name__ == '__main__':
    main()
