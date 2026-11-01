import time
import torch

assert torch.cuda.is_available(), "cuda is not available"
device=torch.device("cuda:0")
N=4096
a=torch.randn(N,N,device=device)
b=torch.randn(N,N,device=device)
#warm up
for _ in range(10):
    c=a@b
torch.cuda.synchronize()
#A:cpu timer without sync
start=time.perf_counter()
c=a@b
t_a=(time.perf_counter()-start)*1000
#B:cpu timer with sync
start=time.perf_counter()
c=a@b
torch.cuda.synchronize()
t_b=(time.perf_counter()-start)*1000
#C:cuda event timer
start_event=torch.cuda.Event(enable_timing=True)
end_event=torch.cuda.Event(enable_timing=True)
start_event.record()
c=a@b
end_event.record()
torch.cuda.synchronize()
t_c=start_event.elapsed_time(end_event)
print(f"cpu timer without sync: {t_a:.3f} ms")
print(f"cpu timer with sync: {t_b:.3f} ms")
print(f"cuda event timer: {t_c:.3f} ms")