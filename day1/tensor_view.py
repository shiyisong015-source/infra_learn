# view只是改变了解读数据的方式但和原始数据共享内存，一般会在不改变数据的情况下改变数据的形状。view()函数返回一个新的tensor，它与原始tensor共享相同的数据，但具有不同的形状。
# view会导致non-contiguous(Taking a view of contiguous tensor could potentially produce a non-contiguous tensor)
import torch
base=torch.tensor([[1,2,3],[4,5,6]])
print(base.is_contiguous())
t=base.transpose(0,1)
print(t.is_contiguous())
# to get a contiguous tensor, we can use contiguous() method
t_contiguous=t.contiguous()
print(t_contiguous.is_contiguous())
#函数permute返回一个view，维度重新排列permute(*dims)
x=torch.randn(2,3,5)
print(f'{x.size()} \n' )
y=x.permute(2,0,1)
print(f'{y.size()} \n' )
# 函数transpose(dim0,dim1)返回一个tensor，交换两个维度
# 如果是strided tensor则返回一个view，如果是一个sparse tensor则返回一个新的tensor
# 当你交换稀疏矩阵的行、列时，存储布局也会跟着切换：CSR ↔ CSC，BSR ↔ BSC。
# 步幅张量是通过"起始位置+各维度的步幅”，确定元素在内存中位置的张量
print(x.stride())#(15,5,1)
# view 返回一个新的tensor共享数据但是不一样的形状，"能否 view"= 是否存在一组新 stride st' = (st'0, ..., st'_{m-1})，使得对任意逻辑下标 (j0,...,j_{m-1})，用新 stride 算出的物理偏移，等于原来同一元素用旧 stride 算出的物理偏移
x=torch.randn(4,4)
x.view(-1,8)
a=torch.randn(1,2,3,4)
b=a.transpose(1,2)
print(a.stride())#(24,12,4,1)
print(b.stride())#(24,4,12,1)
c=a.view(1,3,2,4)
# b,c形状一样但是stride不一样，所以b！=c，transpose是在转置视角，c这是在分块视角
print(f'c {c.stride()}')
print(f'b {b.stride()}')
# view(dtype)把同一块内存的字节按另一种类型重新解释,会对最后一维度进行缩放
y=x.view(torch.int32)

# reshape尽可能返回view实在不行的话进行复制