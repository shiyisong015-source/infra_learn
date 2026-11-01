import torch
x=torch.arange(4)
# expand返回一个新的tensor view，通过修改逻辑shape和stride，让多个逻辑位置映射到同一个物理存储位置
y=x.expand(1000,4)
print("x.shape:",x.shape)
print("x.stride:",x.stride())
print("y.shape:",y.shape)
print("t.stride",y.stride())
print("shared storage:",x.untyped_storage().data_ptr()==y.untyped_storage().data_ptr())
