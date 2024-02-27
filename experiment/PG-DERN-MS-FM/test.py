import torch
import torch.nn.functional as F

from chem_lib.models.encoder import attention

# # 创建两个输入矩阵
# matrix1 = torch.randn(1, 20, 300)
# matrix2 = torch.randn(1, 20, 300)

# 计算注意力分数
# t1 = torch.randn(1,300)
# t2 = torch.randn(1,300)
# t = torch.cat((t1,t2),dim=0)
# x = F.softmax(torch.transpose(t, 1, 0))
# # print(x)
# t = torch.tensor([[1,2],[2,8]],dtype=float)
# w = F.softmax(t)
# print(w)
t1 = torch.tensor([[1,2,3],[1,2,1]])
t2 = torch.tensor([[4,5,6],[1,1,1]])
a = torch.tensor([[1],[1]])
b = torch.tensor([[1],[1]])
print(a*t1+b*t2)