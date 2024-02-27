import json
import time
from operator import itemgetter

import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE

from chem_lib.models import Meta_Trainer, ContextAwareRelationNet
from parser import get_args


root_dir = '.'
args = get_args(root_dir)


model = ContextAwareRelationNet(args)
model = model.to(args.device)
trainer = Meta_Trainer(args, model)

path = 'chem_lib/model_gin/sider.pth'
state = torch.load(path)
model.load_state_dict(state)
start = time.time()
f_tasks = trainer.get_ftask(0)
ts = trainer.test_sars(0, f_tasks)
end = time.time()
print((end-start)/60)
# with open('tssider.json', 'w') as json_file:
#     for key, tensors in ts.items():
#         json.dump({key: tensor.tolist() for key, tensor in tensors.items()}, json_file)
# print(ts)


# with open('tssider.json') as file:
#     db = json.load(file)
#     ys = db['ys']
#     yt = db['yt']
#
#     dic = {}
#     for i in range(len(ys)):
#         dic[i]=ys[i]
#
#     sorted_dict = dict(sorted(dic.items(), key=itemgetter(1),reverse=True))
#     key_list = []
#     for key in sorted_dict:
#         key_list.append(key)
#     print(key_list)

# 226, 280, 275, 287, 1027, 1324, 282, 995, 1340, 455
# 529, 1104, 1492, 1630, 849, 1591, 704, 1088, 1446, 1121
# Allopurinol, Esomeprazole,Lansoprazole,Omeprazole,Dexlansoprazole,Pentostatin,Rabeprazole,Dapiprazole,Dacarbazine,Pantoprazole

# proton pump inhibitor: Esomeprazole,Lansoprazole,Omeprazole,Dexlansoprazole,Rabeprazole,Pantoprazole
# pantoprazole, lansoprazole, omeprazole, esomeprazole, and rabeprazole
# Proton Pump Inhibitor Exposure and Acute Myocardial Infarction Risk: A Nested Cohort Study

# side effect: Esomeprazole,Omeprazole,Dexlansoprazole,Pantoprazole
# Allopurinol: Purine versus non-purine xanthine oxidase inhibitors against cyclophosphamide-induced cardiac and bone marrow toxicity in rats
# Pentostatin: Cardiotoxicity of chemotherapeutic agents: incidence, treatment and prevention

# cyclophosphamide, ifosfamide, cisplatin, carmustine, busulfan, chlormethine , mitomycin
# paclitaxel, etoposide, teniposide, the vinca alkaloids, fluorouracil, cytarabine, amsacrine, cladribine, asparaginase,
# tretinoin pentostatin.

def painting(y_score,q_feats,sider_y_true,sider_q_feats,eval_q_feats):

    # zero_indices = torch.nonzero(sider_y_true == 0)
    # ones_indices = torch.nonzero(sider_y_true == 0)
    # data0 = sider_q_feats[zero_indices[:, 0], zero_indices[:, 1]]
    # data1 = sider_q_feats[ones_indices[:, 0], ones_indices[:, 1]]

    X = torch.cat((q_feats,eval_q_feats),dim=0)
    X = X.detach().cpu().numpy()
    # sider_X = sider_q_feats.detach().cpu().numpy()
    probabilities = y_score.detach().cpu().numpy()
    # y_true = sider_y_true.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # 绘制散点图，根据预测概率赋予不同的颜色
    colors = ['#fa7f6f', '#82b0d2']
    # 定义渐变色列表
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    plt.figure(dpi=601)
    # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=probabilities, cmap='viridis', marker='o', s=50,
    #                       edgecolors='w')
    scatter = plt.scatter(X_tsne[:1704, 0], X_tsne[:1704, 1],marker='o', c=probabilities,cmap=custom_cmap,s=30)
    pos_list = [1714, 1707, 1719, 1704, 1712, 1711, 1705, 1708, 1715, 1717, 1713]

    for i in pos_list:
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], marker='*', c="#d8383a" ,s=100)

    # list = [i for i in range(1704,1720)]
    # for i in list:
    #     plt.annotate(f'P{i}', (X_tsne[i,0], X_tsne[i,1]), textcoords="offset points", xytext=(0, 10), ha='center')
    # 添加颜色条
    legend = plt.colorbar(scatter)
    legend.set_label('Probabilities')
    plt.xticks([])
    plt.yticks([])
    # plt.title('t-SNE Visualization with Predicted Probabilities')
    # plt.show()
    plt.savefig('Probability.png')
def test1():
    pos_list = [1714, 1707, 1719, 1704, 1712, 1711, 1705, 1708, 1715, 1717, 1713]
    for i in range(11):
        pos_list[i] = pos_list[i]-1704
    print(pos_list)
    # 10, 3, 15, 0, 8, 7, 1, 4, 11, 13, 9
# if __name__ == '__main__':
    # root_dir = '.'
    # args = get_args(root_dir)
    # #
    # model = ContextAwareRelationNet(args)
    # model = model.to(args.device)
    # trainer = Meta_Trainer(args, model)
    #
    # path = 'chem_lib/model_gin/sider.pth'
    # state = torch.load(path)
    # model.load_state_dict(state)
    # f_tasks = trainer.get_ftask(0)
    # y_score,y_true,q_feats = trainer.test_sars(0, f_tasks)
    # # torch.save(y_true, 'eval_y_true.pt')
    # torch.save(q_feats, 'eval_q_feats.pt')

    # y_score = torch.load('y_score.pt')
    # q_feats = torch.load('q_feats.pt')
    # sider_y_true = torch.load('sider_y_true.pt')
    # sider_q_feats = torch.load('sider_q_feats.pt')
    # eval_q_feats = torch.load('eval_q_feats.pt')
    # painting(y_score,q_feats,sider_y_true,sider_q_feats,eval_q_feats)

    # test1()