import random
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
from torchmetrics.functional import auroc

import task_generator
from chem_lib.datasets import MoleculeDataset
from encoder import GNN_Encoder, GCN2

from parser import get_args
warnings.filterwarnings("ignore")

# Hyper Parameters
EPISODE = 5000
TEST_EPISODE = 600
LEARNING_RATE = 0.001
GPU = 0

# class RelationNetwork(nn.Module):
#     """docstring for RelationNetwork"""
#     def __init__(self,input_size,hidden_size):
#         super(RelationNetwork, self).__init__()
#         self.layer1 = nn.Sequential(
#                         nn.Conv1d(1,32,kernel_size=5,padding=1),
#                         nn.BatchNorm1d(32, momentum=1, affine=True),
#                         nn.ReLU(),
#                         nn.MaxPool1d(5))
#         self.layer2 = nn.Sequential(
#                         nn.Conv1d(32,32,kernel_size=5,padding=1),
#                         nn.BatchNorm1d(32, momentum=1, affine=True),
#                         nn.ReLU(),
#                         nn.MaxPool1d(5))
#         self.fc0 = nn.Linear(736, 500)
#         self.fc1 = nn.Linear(500,200)
#         self.fc2 = nn.Linear(200,1)
#
#     def forward(self,x):
#         x = torch.reshape(x,(-1,1,x.shape[1]))
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0),-1)
#         out = F.relu(self.fc0(out))
#         out = F.relu(self.fc1(out))
#         out = F.sigmoid(self.fc2(out))
#         return out

class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_sizes, batch_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)
        self.lstm = nn.LSTM(input_size=self.vector_dim,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True)
    def forward(self, inputs):
        # c0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),
        #               requires_grad=False).cuda()
        # h0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),
        #               requires_grad=False).cuda()
        output, _ = self.lstm(inputs)
        return output

class DistanceNetwork(nn.Module):
    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):

        """
        Produces pdfs over the support set classes for the target set image.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_magnitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities

class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):

        """
        Produces pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarities of size [sequence_length, batch_size]
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image
                                                                            [sequence_length,  batch_size, num_classes]
        :return: Softmax pdf
        """
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        return preds

class MatchingNetwork(nn.Module):

    def __init__(self, input_size,hidden_size):
        super(MatchingNetwork, self).__init__()
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
        self.g = nn.Sequential(self.layer1,self.layer2)
        self.lstm = BidirectionalLSTM(layer_sizes=[352], batch_size=21, vector_dim = 352)
        # self.dn = DistanceNetwork()

    def forward(self, x):
        # produce embeddings for support set images
        x = torch.reshape(x, (-1, 1, x.shape[1]))
        z = self.g(x)
        z = z.view(z.size(0), -1)
        support = z[:2 * 10].view(2, 10, -1).mean(1)
        query = z[2 * 10:]
        all_similarity = []
        # produce embeddings for target images
        for i in np.arange(query.size(0)):
            gen_encode = query[i].view(1,-1)
            outputs = torch.cat((support,gen_encode),dim=0).unsqueeze(0)

            outputs = self.lstm(outputs).squeeze(0)

            # get similarity between support set embeddings and target
            # similarities = self.dn(support_set=outputs[:-1], input_image=outputs[-1])
            similarities = F.cosine_similarity(outputs[:-1].view(2,-1),outputs[-1].view(1,-1))
            all_similarity.append(similarities)
        all_similarity = torch.stack(all_similarity,dim=0) # 16*2
        preds = F.softmax(all_similarity,dim=1) # 16*2
        return preds

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


def main():
    sys.stdout = Logger('resultmn.txt')
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
    relation_network = MatchingNetwork(300,128)
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

    if os.path.exists(str("./models/mn/feature_encoder_" + str(9) +"way_" + str(10) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/mn/feature_encoder_" + str(9) +"way_" + str(10) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/mn/relation_network_"+ str(9) +"way_" + str(0) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/mn/matching_network_"+ str(10) +"way_" + str(0) +"shot.pkl")))
        print("load relation network success")
    if os.path.exists(str("./models/mn/message_gcn_"+ str(9) +"way_" + str(0) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/mn/message_gcn_"+ str(10) +"way_" + str(0) +"shot.pkl")))
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

            # node_s_emb =node_s_emb.view(2,10,300)
            # sub_s_emb = sub_s_emb.view(2,10,300)
            # s_emb = s_emb.view(2,10,300)
            #
            # node_s_emb = torch.sum(node_s_emb, 1).squeeze(1)
            # sub_s_emb = torch.sum(sub_s_emb, 1).squeeze(1)
            # s_emb = torch.sum(s_emb, 1).squeeze(1)


            # calculate relations
            # each batch sample link to every samples to calculate relations
            # to form a 100x128 matrix for relation network
            support_features_ext = node_s_emb
            query_features_ext = node_q_emb

            labels = query_labels

            relation_pairs = torch.cat((support_features_ext,query_features_ext),0).view(-1,300)
            relations1 = relation_network(relation_pairs).view(-1,2)

            support_features_ext = sub_s_emb
            query_features_ext = sub_q_emb

            relation_pairs = torch.cat((support_features_ext,query_features_ext),0).view(-1,300)
            relations2 = relation_network(relation_pairs).view(-1,2)

            support_features_ext = s_emb
            query_features_ext = q_emb

            relation_pairs = torch.cat((support_features_ext,query_features_ext),0).view(-1,300)
            relations3 = relation_network(relation_pairs).view(-1,2)

            # mse = nn.MSELoss().cuda(GPU)
            l1loss = nn.L1Loss().cuda(GPU)
            ce = nn.CrossEntropyLoss().cuda(GPU)
            # one_hot_labels = torch.zeros(16, 2).cuda(GPU).scatter_(1, query_labels.view(-1, 1), 1)
            # loss = mse(relations3,one_hot_labels)
            loss = ce(relations3,query_labels.view(-1).long())
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

                    # cache_s_emb = cache_s_emb.view(2, 10, 300)
                    # cache_s_emb = torch.sum(cache_s_emb, 1).squeeze(1)

                    support_features_ext = cache_s_emb
                    query_features_ext = q_emb

                    relation_pairs = torch.cat((support_features_ext, query_features_ext), 0).view(-1, 300)
                    relations = relation_network(relation_pairs).view(-1, 2)

                    predict = relations.detach()[:,1]
                    predict_labels.append(predict)

                y_true = torch.cat(y_true_list, 0)
                prediction = torch.cat(predict_labels, 0)
                auc = auroc(prediction, y_true, pos_label=1).item()
                auc_scores.append(auc)

            avg_scores = np.mean(auc_scores)
            end1 = time.time()
            print("avg_scores:",avg_scores,"t1:",auc_scores[0],'t2:',auc_scores[1],"t3:",auc_scores[2],"use:",str(end1 - start1))

            if avg_scores > last_scores:
                # save networks
                torch.save(feature_encoder.state_dict(),str("./models/mn/feature_encoder_" + str(9) +"way_" + str(10) +"shot.pkl"))
                torch.save(relation_network.state_dict(),str("./models/mn/matching_network_"+ str(9) +"way_" + str(10) +"shot.pkl"))
                torch.save(message_gcn.state_dict(),str("./models/mn/message_gcn_" + str(9) + "way_" + str(10) + "shot.pkl"))
                print("save networks for episode:",episode)
                last_scores = avg_scores

if __name__ == '__main__':
    main()
