import statistics
import torch
assert torch.cuda.is_available(),"cuda is not available"
device=torch.device("cuda:0")
dtype=torch.float32
sizes_mib=[4**i for i in range(1,5) ]
repeats=30
print("GPU:",torch.cuda.get_device_name(0))
def benchmark(fn):
    for _ in range(15):
        fn()
    torch.cuda.synchronize()
    times=[]
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)
with torch.no_grad():
    for size_mib in sizes_mib:
        n=size_mib*1024*1024//4
        x=torch.randn(n,device=device,dtype=dtype)
        y=torch.randn(n,device=device,dtype=dtype)
        z=torch.empty_like(x)
        copy_ms=benchmark(lambda:z.copy_(x))
        add_ms=benchmark(lambda: torch.add(x,y,out=z))
        copy_gb=2*x.numel()*x.element_size()/1e9
        add_gb=3*x.numel()*x.element_size()/1e9
        copy_bw=copy_gb/(copy_ms*1e-3)
        add_bw=add_gb/(add_ms*1e-3)
        print(
            f"size:{size_mib:4d} MiB |"
            f"copy:{copy_ms:.3f}ms,{copy_bw:.3f} GB/s |"
            f"add:{add_ms:.3f}ms,{add_bw:.3f} GB/s"
        )