import json
import random

import torch
from numpy import mean
from rdkit.Chem import AllChem
from torch_geometric.data import DataLoader

from chem_lib.datasets.loader import mol_to_graph_data_obj_simple
from chem_lib.models.encoder import GNN_Encoder, GCN
from parser import get_args
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

root_dir = '.'
args = get_args(root_dir)

gcn = GCN(args.emb_dim, args.emb_dim)
gcn = gcn.to(args.device)

def loader_batch(data):
    loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
    for samples in loader:
        samples = samples.to(args.device)
        return samples
def get_mol_list(path,num):
    smile_list = []
    with open(path) as pos_file:
        binary_list = json.load(pos_file)
    for i in binary_list[1]:
        smile_list.append(i)
    if num > 0:
        smile_list = random.sample(smile_list,num)
    mol_list = [AllChem.MolFromSmiles(s) for s in smile_list]
    return mol_list
def get_graph_data(mol_list):
    data_list = []
    for mol in mol_list:
        graph_data = mol_to_graph_data_obj_simple(mol)
        data_list.append(graph_data)
    return data_list

def ftask_aug(pos_emb, f_embs):
    fc_embs = torch.cat((pos_emb, f_embs), dim=0)
    x = fc_embs / torch.norm(fc_embs, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模
    similarity = torch.mm(x, x.T)  # 矩阵乘法
    new_pos_emb = gcn(similarity, fc_embs)
    pos_aug_emb= torch.chunk(new_pos_emb, 4, dim=0)[0]
    return pos_aug_emb
def saveemb(args):
    model = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                        drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                        batch_norm=args.enc_batch_norm)
    model = model.to(args.device)
    model.gnn.load_state_dict(torch.load(args.pretrained_weight_path, map_location='cuda:0'))
    sample_path = "data/tox21/new/"
    pos_json = sample_path+"11/raw/tox21.json"
    f1_json = sample_path+"3/raw/tox21.json"
    f2_json = sample_path+"5/raw/tox21.json"
    f3_json = sample_path+"8/raw/tox21.json"

    pos_list = []   # SR-MMP 11
    # case_list = ['Oc1ccc(-c2ccccc2)c(Cl)c1','O=C1c2ccccc2C(=O)C1(O)O','CC1=CC(C)(C)Nc2ccccc21','Oc1ccc(Cc2ccccc2O)cc1','COc1ccc(C(=O)c2ccccc2O)c(O)c1']  # SR-MMP
    case_list = []
    f1_list = []    # NR-AhR 3
    f2_list = []    # NR-ER 5
    f3_list = []    # SR-ARE 8

    pos_list = get_mol_list(pos_json,-1)
    # case_list = [AllChem.MolFromSmiles(s) for s in case_list]
    f1_list = get_mol_list(f1_json,5)
    f2_list = get_mol_list(f2_json,5)
    f3_list = get_mol_list(f3_json,5)

    pos_data = get_graph_data(pos_list)
    f1_data = get_graph_data(f1_list)
    f2_data = get_graph_data(f2_list)
    f3_data = get_graph_data(f3_list)

    pos_loader = loader_batch(pos_data)
    f1_loader = loader_batch(f1_data)
    f2_loader = loader_batch(f2_data)
    f3_loader = loader_batch(f3_data)

    pos_emb,pos_emb,pos_emb = model(pos_loader)
    f1_emb,f1_emb,f1_emb = model(f1_loader)
    f2_emb,f2_emb,f2_emb = model(f2_loader)
    f3_emb,f3_emb,f3_emb = model(f3_loader)
    f_embs = torch.cat((f1_emb,f2_emb,f3_emb),dim=0)

    torch.save(pos_emb,'data/posemb.pt')
    torch.save(f_embs,'data/fembs.pt')

def getSimMat():
    model = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                        drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                        batch_norm=args.enc_batch_norm)
    liner = torch.nn.Linear(300,128)
    model = model.to(args.device)
    liner = liner.to(args.device)
    model.gnn.load_state_dict(torch.load(args.pretrained_weight_path, map_location='cuda:0'))
    smile_list = []
    mol1 = 'CC(CC1=CC=CC=C1)N(C)CC2=CC=CC=C2'
    mol2 = 'CC(COC1=CC=CC=C1)N(CCCl)CC2=CC=CC=C2'
    mol3 = 'C=C1[C@H](C[C@@H]([C@H]1CO)O)N2C=NC3=C2NC(=NC3=O)N'
    # 'C1=COC(=C1)CNC2=NC=NC3=C2NC=N3'
    # 'C[C@@H](C(=O)N1CC2=CC=CC=C2C[C@H]1C(=O)O)N[C@@H](CCC3=CC=CC=C3)C(=O)O'
    # 'C1CCC(CC1)[C@@H]2C[C@H](N(C2)C(=O)CP(=O)(CCCCC3=CC=CC=C3)O)C(=O)O'
    smile_list.append(mol1)
    smile_list.append(mol2)
    smile_list.append(mol3)
    mol_list = [AllChem.MolFromSmiles(s) for s in smile_list]
    mol_graphs = get_graph_data(mol_list)
    mol_loader = loader_batch(mol_graphs)
    mol_embs,mol_embs,mol_embs = model(mol_loader)
    mol_embs = liner(mol_embs)
    x = mol_embs/torch.norm(mol_embs,dim=-1,keepdim=True)
    adj = torch.mm(x,x.T)
    print(adj)

# if __name__ == '__main__':
#     getSimMat()
# saveemb(args)

if __name__ == '__main__':
    pos_emb = torch.load('data/posemb.pt')
    f_embs = torch.load('data/fembs.pt')

    # select
    # plt.figure(dpi=500)
    # tsne = TSNE(n_components=2,init='pca',random_state=0,learning_rate='auto')
    # X = pos_emb.cpu().detach().numpy()
    # X_tsne = tsne.fit_transform(X)
    # x1 = [X_tsne[i,0] for i in range(918)]
    # y1 = [X_tsne[i,1] for i in range(918)]
    # plt.scatter(x1, y1, s=5, c='blue', marker='o')
    # for i in range(0,918,10):
    #     plt.annotate("{}".format(i),xy=(x1[i],y1[i]))
    # plt.show()

    origin = []
    other = []
    ori_list = [750,630,660,470,560]
    # ori_list = [random.randint(0,917)for i in range(5)]
    for i in range(918):
        if i in ori_list:
            origin.append(pos_emb[i])
        else:
            other.append(pos_emb[i])

    origin_emb = torch.stack(origin,dim=0)
    other_emb = torch.stack(other,dim=0)
    aug_emb = ftask_aug(origin_emb,f_embs)

    # 原始图
    plt.figure(dpi=601)
    tsne = TSNE(n_components=2,init='pca',random_state=0,learning_rate='auto')
    # X = torch.cat((origin_emb,other_emb),dim=0).cpu().detach().numpy()
    # X_tsne = tsne.fit_transform(X)
    # x1 = [X_tsne[i,0] for i in range(5)]
    # y1 = [X_tsne[i,1] for i in range(5)]
    # x2 = [X_tsne[i,0] for i in range(5,918)]
    # y2 = [X_tsne[i,1] for i in range(5,918)]
    # zx1 = [X_tsne[i,0] for i in range(918)]
    # zy1 = [X_tsne[i,1] for i in range(918)]
    # plt.scatter(x2, y2, s=5, c='#298091', marker='o', alpha=0.8, label = "unknown_data")
    # plt.scatter(x1, y1, s=20, c='#f37c21', marker='o', label = "original_data")
    # plt.scatter(mean(zx1), mean(zy1), s=100, c='#9b2d61', marker='*',label="center")
    # plt.legend(loc="upper right",fontsize=15)
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig('data/out1.png')
    # plt.show()
    #
    # # 增强图

    # plt.figure(dpi=500)
    # plt.figure(dpi=601)
    # tsne = TSNE(n_components=2, init='pca', random_state=0, learning_rate='auto')
    Z = torch.cat((aug_emb,other_emb),dim=0).cpu().detach().numpy()  # n*D
    Z_tsne = tsne.fit_transform(Z) # n*2
    x3 = [Z_tsne[i,0] for i in range(5)]
    y3 = [Z_tsne[i,1] for i in range(5)]
    x4 = [Z_tsne[i,0] for i in range(5,918)]
    y4 = [Z_tsne[i,1] for i in range(5,918)]
    zx2 = [Z_tsne[i,0] for i in range(918)]
    zy2 = [Z_tsne[i,1] for i in range(918)]
    plt.scatter(x4, y4,s=5,c='#298091',marker='o',alpha=0.8,label = "unknown_data")
    plt.scatter(x3, y3, s=20, c='#f37c21', marker='o',label = "augmented_data")
    plt.scatter(mean(zx2), mean(zy2), s=100, c='#9b2d61', marker='*',label="center")
    plt.legend(loc="upper right",fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('data/out2.png')
    plt.show()
#
#
#
#
#
#
#
#     # # 增强图
#     # plt.figure(dpi=500)
#     # tsne = TSNE(n_components=2,init='pca',random_state=0,learning_rate='auto')
#     # Z = torch.cat((case_aug_emb,pos_emb),dim=0).cpu().detach().numpy()
#     # Z_tsne = tsne.fit_transform(Z)
#     # x3 = [Z_tsne[i,0] for i in range(5)]
#     # y3 = [Z_tsne[i,1] for i in range(5)]
#     # x4 = [Z_tsne[i,0] for i in range(5,918)]
#     # y4 = [Z_tsne[i,1] for i in range(5,918)]
#     # plt.scatter(x4,y4,c='#5071c3', s=5,marker='o',label = 'positive')
#     # plt.scatter(x3,y3,c='#ff9518',s=20,marker='*',label = 'augment')
#     # plt.legend(loc="upper right")
#     # plt.xticks([])
#     # plt.yticks([])
#     # plt.savefig('data/out2.png')
#     # plt.show()
