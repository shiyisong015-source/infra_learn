import torch 
# from data
data=[[1,2],[3,4]]
x_data=torch.tensor(data)
# from numpy
import numpy as np
np_array=np.array(data)
x_np=torch.from_numpy(np_array)
# from another tensor
x_ones=torch.ones_like(x_data)
print(f'ones tensor:\n{x_ones} \n')
x_rand=torch.rand_like(x_data,dtype=torch.float)
print(f'random tensor:\n{x_rand} \n')
# shape is a tuple of tensor dimensions
shape=(2,3)
rand_tensor=torch.rand(shape)
ones_tensor=torch.ones(shape)
zeros_tensor=torch.zeros(shape)
print(f'random tensor:\n{rand_tensor} \n')
print(f'ones tensor:\n{ones_tensor} \n')
print(f'zeros tensor:\n{zeros_tensor} \n')
# tensor attributes dtype shape device
tensor=torch.rand(3,4)
print(f'tensor:\n{tensor} \n')
print(f'dtype: {tensor.dtype} \n')
print(f'shape: {tensor.shape} \n')          
print(f'device: {tensor.device} \n')
# operations on tensors
if torch.cuda.is_available():
    tensor=tensor.to('cuda')
    print(f'device: {tensor.device} \n')
# tensor operations
# indexing and slicing
tensor=torch.ones(4,4)
print(f'first row:{tensor[0]}\n')
print(f'first column:{tensor[:,0]}\n')
print(f'last column:{tensor[...,-1]} \n')
tensor[:,1]=0
print(tensor)
t1=torch.cat([tensor,tensor,tensor],dim=1)
print(t1)
# 
