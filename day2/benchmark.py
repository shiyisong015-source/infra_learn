import time
import statistics
import torch
assert torch.cuda.is_available(), "cuda is not available"
device=torch.device("cuda:0")
N=4096
REPEATS=30
print("GPU:",torch.cuda.get_device_name(0))
print("pytorch:",torch.__version__)
a=torch.randn(N,N,device=device,dtype=torch.float32)
b=torch.randn(N,N,device=device,dtype=torch.float32)
out=torch.empty_like(a)
def matmul():
    torch.mm(a,b,out=out)
with torch.no_grad():
    for _ in range(15):
        matmul()
    torch.cuda.synchronize()
    
    cpu_times=[]
    event_times=[]
    
    for _ in range(REPEATS):
        #cpu timer+device synchronization
        start=time.perf_counter()
        matmul()
        torch.cuda.synchronize()
        cpu_times.append((time.perf_counter()-start)*1000)
        
        start=torch.cuda.Event(enable_timing=True)
        start.record()
        matmul()
        end=torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        event_times.append(start.elapsed_time(end))
        
def report(name,times):
    print(f"\n{name}")
    print(f"mean:{statistics.median(times):.3f} ms")
    print(f"min:{min(times):.3f} ms")
    print(f"max:{max(times):.3f} ms")
    print(f"std:{statistics.stdev(times):.3f} ms")
    
report("cpu timer+device synchronization",cpu_times)
report("cuda event timer",event_times)
latency_ms=statistics.median(event_times)
flops=2*N**3
tflops=flops/(latency_ms*1e-3)/1e12
print(f"\nFLOPS: {flops:.3e}")