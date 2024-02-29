import json

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc



with open('tsdern.json') as jsonfile:
    db = json.load(jsonfile)
    ys1 = db['0']['ys']
    yt1 = db['0']['yt']
with open('tsmetasgd.json') as jsonfile:
    db = json.load(jsonfile)
    ys2 = db['0']['ys']
    yt2 = db['0']['yt']
with open('tsfomaml.json') as jsonfile:
    db = json.load(jsonfile)
    ys3 = db['0']['ys']
    yt3 = db['0']['yt']
with open('tsrn.json') as jsonfile:
    db = json.load(jsonfile)
    ys4 = db['0']['ys']
    yt4 = db['0']['yt']
with open('tspn.json') as jsonfile:
    db = json.load(jsonfile)
    ys5 = db['0']['ys']
    yt5 = db['0']['yt']
with open('tsmn.json') as jsonfile:
    db = json.load(jsonfile)
    ys6 = db['0']['ys']
    yt6 = db['0']['yt']


plt.figure(dpi=800)
fpr1, tpr1, threshold1 = roc_curve(yt1, ys1)
roc_auc1 = auc(fpr1,tpr1)   # 准确率代表所有正确的占所有数据的比值
fpr2, tpr2, threshold2 = roc_curve(yt2, ys2)
roc_auc2 = auc(fpr2,tpr2)   # 准确率代表所有正确的占所有数据的比值
fpr3, tpr3, threshold3 = roc_curve(yt3, ys3)
roc_auc3 = auc(fpr3,tpr3)   # 准确率代表所有正确的占所有数据的比值
fpr4, tpr4, threshold4 = roc_curve(yt4, ys4)
roc_auc4 = auc(fpr4,tpr4)   # 准确率代表所有正确的占所有数据的比值
fpr5, tpr5, threshold5 = roc_curve(yt5, ys5)
roc_auc5 = auc(fpr5,tpr5)   # 准确率代表所有正确的占所有数据的比值
fpr6, tpr6, threshold6 = roc_curve(yt6, ys6)
roc_auc6 = auc(fpr6,tpr6)   # 准确率代表所有正确的占所有数据的比值

lw = 2
plt.subplot(1,1,1)
plt.plot(fpr1, tpr1, color='darkorange',lw=lw, label='MAML-PG-DERN (%0.4f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='#8ecfc9',lw=lw, label='MS-PG-DERN (%0.4f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='#ffbe7a',lw=lw, label='FM-PG-DERN (%0.4f)' % roc_auc3)
plt.plot(fpr4, tpr4, color='#fa7f6f',lw=lw, label='RN-PG-DERN (%0.4f)' % roc_auc4)
plt.plot(fpr5, tpr5, color='#82b0d2',lw=lw, label='PN-PG-DERN (%0.4f)' % roc_auc5)
plt.plot(fpr6, tpr6, color='#beb8dc',lw=lw, label='MN-PG-DERN (%0.4f)' % roc_auc6)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity',fontsize=15)
plt.ylabel('Sensitivity',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.title('ROC_AUC',fontsize=14)
plt.legend(loc="lower right",fontsize=15)
# plt.show()
plt.tight_layout()
plt.savefig('task02.png')


# plt.figure(dpi=800)
# # categories = ['maml', 'meta_sgd', 'fomaml', 'relation network','prototypical network','matching network']
# # values = [0.8525, 0.8439, 0.8302, 0.7242, 0.6774, 0.6716]
# # colors = ['darkorange', '#8ecfc9', '#ffbe7a', '#fa7f6f','#82b0d2','#beb8dc']
# categories = ['matching network', 'prototypical network', 'fomaml', 'relation network','meta_sgd','maml']
# values = [0.6716, 0.6774, 0.7242, 0.8302, 0.8439, 0.8525]
# colors = ['#beb8dc', '#82b0d2', '#fa7f6f', '#ffbe7a','#8ecfc9','darkorange']
#
# barheight = 0.5
# # 绘制横向柱状图
# plt.barh(categories, values,color=colors,height=barheight)
#
# plt.xlim([0.6, 0.9])
# # 添加标题和标签
# # plt.title('avg_auc')
# plt.xlabel('avg_auc')
# plt.ylabel('type')
#
# # 显示图表
# plt.show()



# 柱状图
# categories = ['Matching Network', 'Prototypical Network', 'Relation Network', 'FOMAML','Meta-SGD','MAML']
# values = [0.6716, 0.6774, 0.7242, 0.8302, 0.8439, 0.8525]
# colors = ['#beb8dc', '#82b0d2', '#fa7f6f', '#ffbe7a','#8ecfc9','darkorange']
# # 绘制横向柱状图
# fig, ax = plt.subplots()
#
# # 遍历每一行数据，创建横向柱状图
# for i, category in enumerate(categories):
#     bars = ax.barh(category, values[i], color=colors[i],label=category,height=0.5)
#     for bar in bars:
#         yval = bar.get_width()
#         ax.text(yval, bar.get_y() + bar.get_height() / 2, round(yval, 4),va='center', ha='left',fontsize=15)
#
# # 添加图例
# ax.legend(loc="lower right",fontsize=15)
# ax.yaxis.set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xlim([0.6, 0.9])
# plt.xticks(fontsize=15)
# plt.tight_layout()
# # 添加标题和标签
# plt.savefig('avg1.png')
