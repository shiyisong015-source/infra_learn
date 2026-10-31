import torch
def inspect(name,x):
    print(f'\n==={name}===')
    print("shape:",tuple(x.shape))
    print("stride:",x.stride())
    print("dtype:",x.dtype)
    print("contiguous:",x.is_contiguous())
    print("logical nbytes:",x.numel()*x.element_size())
    print("data_ptr:",x.data_ptr())

# transpose
x=torch.arange(12,dtype=torch.float32).reshape(3,4)
y=x.transpose(0,1)
z=y.contiguous()
inspect("original x",x)
inspect("transposed y",y)
inspect("contiguous z",z)
# untyped_storage（）获取tensor的底层存储对象（按字节存放的底层容器），tensor分为元数据和底层存储
print("\n Storage sharing:",x.untyped_storage().data_ptr()==y.untyped_storage().data_ptr())
# expriment 2: view vs reshape
print("\n ===view vs reshape===")
try:
    v=y.view(12)
    inspect("view v",v)
except RuntimeError as e:
    print("view failed:",e)
r=y.reshape(12)
inspect("reshape r",r)  
print("Reshape shares storage:",
      r.untyped_storage().data_ptr()
      == y.untyped_storage().data_ptr())
# expriment 3 dtype and memory
print("\n===dtype mrmory===")
for dtype in [torch.float32,torch.float16,torch.bfloat16]:
    t=torch.zeros((1024,1024),dtype=dtype)
    inspect(str(dtype),t)
# expriment 4: kv cache footprint
B=8
L=4096
H_KV=8
D=128
N_LAYERS = 32
BYTES_PER_ELEMENT=2
single_k=B*L*H_KV*D*BYTES_PER_ELEMENT
single_kv=2*single_k
total_kv=N_LAYERS*single_kv
print("\n=== KV cache ===")
print("single-layer K(MiB):",single_k/1024**2)
print("single-layer KV(MiB):", single_kv/1024**2)
print("Total KV(GiB):",total_kv/1024**3)