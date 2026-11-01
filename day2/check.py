
import statistics
import torch

assert torch.cuda.is_available()

N = 4096
REPEATS = 30

torch.manual_seed(42)
a = torch.randn(N, N, device="cuda", dtype=torch.float32)
b = torch.randn(N, N, device="cuda", dtype=torch.float32)

def bench(a, b, repeats=REPEATS):
    out = torch.empty(
        (a.shape[0], b.shape[1]),
        device=a.device,
        dtype=a.dtype,
    )

    for _ in range(15):
        torch.mm(a, b, out=out)
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(repeats):
        start.record()
        torch.mm(a, b, out=out)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    return statistics.median(times), out.clone()


with torch.no_grad():
    # 1. IEEE FP32
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    t_fp32, ref = bench(a, b)

    # 2. TF32 Tensor Core
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    t_tf32, result_tf32 = bench(a, b)

    # 3. BF16 Tensor Core
    a_bf16 = a.to(torch.bfloat16)
    b_bf16 = b.to(torch.bfloat16)
    t_bf16, result_bf16 = bench(a_bf16, b_bf16)

def error(result, reference):
    result = result.float()
    relative = (
        torch.linalg.vector_norm(result - reference)
        / torch.linalg.vector_norm(reference)
    )
    return relative.item()

def tflops(ms):
    return 2 * N**3 / (ms * 1e9)

for name, ms, result in [
    ("FP32", t_fp32, ref),
    ("TF32", t_tf32, result_tf32),
    ("BF16", t_bf16, result_bf16),
]:
    print(
        f"{name:5s} "
        f"Latency: {ms:.3f} ms | "
        f"Throughput: {tflops(ms):.2f} TFLOP/s | "
        f"Relative error: {error(result, ref):.6f}"
    )

torch.backends.cuda.matmul.fp32_precision = "ieee"
