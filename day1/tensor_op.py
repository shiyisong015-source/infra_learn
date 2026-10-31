import torch
import numpy as np
tensor=torch.ones(4,4)
tensor[:,1]=0
tensor=torch.cat([tensor,tensor],dim=1)
# 矩阵计算以下等价，torch.matmul()把结果写入预先分配好的tensor中，可以避免新建张量
y1=tensor @ tensor.T
y2=tensor.matmul(tensor.T)
y3=torch.rand_like(y1)
torch.matmul(tensor,tensor.T,out=y3)
# 逐元素计算
z1=tensor*tensor
z2=tensor.mul(tensor)
z3=torch.rand_like(z1)
torch.mul(tensor,tensor,out=z3)
# item()方法将单元素张量转换为Python数值
agg=tensor.sum()
agg_item=agg.item()
print(f'agg: {agg} , agg_item: {agg_item} \n')
# 原位操作一般后缀为_，例如x.copy_(y), x.t_(), x.add_(5)
print(f'{tensor} \n')
tensor.add_(5)
print(f'{tensor} \n')
# bridge with Numpy
# 在cpu上，tensor何numpy可以共享内存，修改一个会影响另一个
# tensor to numpy array
t=torch.ones(5)
print(f'tensor: {t} \n')
n=t.numpy()
print(f'numpy: {n} \n')
t.add_(1)
print(f'tensor: {t} \n')
print(f'numpy: {n} \n')
# numpy array to tensor
n=np.ones(5)
t=torch.from_numpy(n)
np.add(n,1,out=n)
print(f'tensor: {t} \n')    
print(f'numpy: {n} \n')

