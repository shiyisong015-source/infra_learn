import torch

# Experiment 2: Matrix multiplication
M, K, N = 128, 256, 512

A = torch.randn(M, K)
B = torch.randn(K, N)

C = torch.mm(A, B)
# FLOPs完成一次计算所需的浮点操作数量；FLOPS每秒执行多少浮点操作；latency一次计算执行了多长时间
flops = 2 * M * K * N

print("MatMul output:", C.shape)
print("MatMul GFLOPs:", flops / 1e9)

# Experiment 3: Batched matrix multiplication
batch = 8

A_batch = torch.randn(batch, M, K)
B_batch = torch.randn(batch, K, N)

C_batch = torch.bmm(A_batch, B_batch)

bmm_flops = 2 * batch * M * K * N

print("BMM output:", C_batch.shape)
print("BMM GFLOPs:", bmm_flops / 1e9)

# Experiment 4: matmul broadcasting
A = torch.randn(8, 128, 256)
B = torch.randn(256, 512)

C = torch.matmul(A, B)

print("Broadcast MatMul output:", C.shape)