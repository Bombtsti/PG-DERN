import numpy as np
import matplotlib.pyplot as plt

#分子图
'''
from rdkit import Chem
from rdkit.Chem import Draw
m = Chem.MolFromSmiles("Cc1cnc(C)c(C)n1")
Draw.MolToFile(m, 'data/output.png', size=(150, 150))
'''


rootPath = 'ablation2/'
# ablation 1
#
# list = []
# for i in range(5):
#     file = rootPath+'t{}.txt'.format(i)
#     l = []
#     # cout = [100,150,200,300,400,500]
#     cout = [5,15,30,50,100,200,500]
#     with open(file,'r') as f:
#         lines = f.readlines()
#         line = []
#         for c in cout:
#             line.append(lines[c])
#         for datas in line:
#             data = datas.split()[6]
#             l.append(data)
#     list.append(l)
# a1 = np.array(list,dtype=float)
# mean_a1 = np.mean(a1,axis=0)
#
# list = []
# for i in range(5):
#     file = rootPath+'ts{}.txt'.format(i)
#     l = []
#     # cout = [i for i in range(0,501,50)]
#     cout = [5,15,30,50,100,200,500]
#     with open(file,'r') as f:
#         lines = f.readlines()
#         line = []
#         for c in cout:
#             line.append(lines[c])
#         for datas in line:
#             data = datas.split()[6]
#             l.append(data)
#     list.append(l)
# a2 = np.array(list,dtype=float)
# mean_a2 = np.mean(a2,axis=0)
#
#
# list = []
# for i in range(5):
#     file = rootPath+'tg{}.txt'.format(i)
#     l = []
#     # cout = [i for i in range(0,501,50)]
#     cout = [1,3,6,10,20,40,100]
#     with open(file,'r') as f:
#         lines = f.readlines()
#         line = []
#         for c in cout:
#             line.append(lines[c])
#         for datas in line:
#             data = datas.split()[6]
#             l.append(data)
#     list.append(l)
# a3 = np.array(list,dtype=float)
# mean_a3 = np.mean(a3,axis=0)
#
# plt.figure(figsize=(12,10),dpi=601)
#
# # x1 = [i for i in range(0,5001,500)]
# # x2 = [i for i in range(0,5001,500)]
# # x3 = [i for i in range(0,5001,500)]
# x1 = [1,2,3,4,5,6,7]
# y1 = mean_a1
# y2 = mean_a2
# y3 = mean_a3
#
# plt.plot(x1,y1,label='PG-DERN',color='#529fc4',linestyle='-',linewidth=8,marker='*',markersize=22)
# plt.plot(x1,y2,label='w/o subEncoder',color='#299e8b',linestyle='-',linewidth=8,marker='o',markersize=16)
# plt.plot(x1,y3,label='w/o nodeEncoder',color='#f77912',linestyle='-',linewidth=8,marker='s',markersize=16)
#
# # xticks = [i for i in range(0,5001,500)]
# xticks = [50,150,300,500,1000,2000,5000]
# yticks = [i for i in np.arange(0.60,0.861,0.05)]
# plt.xticks(x1,xticks,fontsize=40)
# plt.yticks(yticks,fontsize=40)
#
# plt.ylim(0.60,0.861)
#
# # plt.title('10 shot')
# plt.xlabel('Epoch',fontsize=45)
# plt.ylabel('AUC',fontsize=45)
# # plt.grid(alpha=0.4)
# plt.legend(loc="lower right",fontsize=40)
# # plt.show()
# plt.tight_layout()
# plt.savefig(rootPath+'ablation1.png')



list = []
for i in range(0,21,5):
    file = rootPath+'t{}.txt'.format(i)
    l = []
    # cout = [100,150,200,300,400,500]
    cout = [5,15,30,50,100,200,500]
    with open(file,'r') as f:
        lines = f.readlines()
        line = []
        for c in cout:
            line.append(lines[c])
        for datas in line:
            data = datas.split()[6]
            l.append(data)
    list.append(l)
a1 = np.array(list,dtype=float)
mean_a1 = np.mean(a1,axis=0)

list = []
for i in range(0,21,10):
    file = rootPath+'td{}.txt'.format(i)
    l = []
    # cout = [i for i in range(0,501,50)]
    cout = [5,15,30,50,100,200,500]
    with open(file,'r') as f:
        lines = f.readlines()
        line = []
        for c in cout:
            line.append(lines[c])
        for datas in line:
            data = datas.split()[6]
            l.append(data)
    list.append(l)
a2 = np.array(list,dtype=float)
mean_a2 = np.mean(a2,axis=0)

plt.figure(figsize=(12,10),dpi=601)

# x1 = [i for i in range(0,5001,500)]
# x2 = [i for i in range(0,5001,500)]
# x3 = [i for i in range(0,5001,500)]
x1 = [1,2,3,4,5,6,7]
y1 = mean_a1
y2 = mean_a2

plt.plot(x1,y1,label='PG-DERN',color='#529fc4',linestyle='-',linewidth=8,marker='*',markersize=22)
# plt.plot(x1,y2,label='w/o subEncoder',color='#299e8b',linestyle='-',linewidth=8,marker='o',markersize=16)
plt.plot(x1,y2,label='w/o disagreement',color='#f77912',linestyle='-',linewidth=8,marker='s',markersize=16)

# xticks = [i for i in range(0,5001,500)]
xticks = [50,150,300,500,1000,2000,5000]
yticks = [i for i in np.arange(0.77,0.851,0.02)]
plt.xticks(x1,xticks,fontsize=40)
plt.yticks(yticks,fontsize=40)

plt.ylim(0.77,0.85)

# plt.title('10 shot')
plt.xlabel('Epoch',fontsize=45)
plt.ylabel('AUC',fontsize=45)
# plt.grid(alpha=0.4)
plt.legend(loc="lower right",fontsize=40)
# plt.show()
plt.tight_layout()
plt.savefig(rootPath+'ablation2.png')

