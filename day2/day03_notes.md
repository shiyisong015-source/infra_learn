# Day 3 — GPU Memory Hierarchy, Bandwidth & Roofline

> 学习项目：LLM Inference Systems · Week 1 / Day 3  
> 实验设备：NVIDIA A100-PCIE-40GB  
> 学习目标：区分 CUDA 内存空间与缓存层次、逻辑有效带宽与真实 HBM 流量，理解 Arithmetic Intensity / Roofline 的用途与局限，以及如何用 GPU 指标继续定位瓶颈。  
> 状态：**复盘修订版**（基于个人答案与讨论整理；尚未用 Nsight Compute 实测 DRAM 流量）

## Part A — 核心概念与订正

### Q1. GPU Memory Hierarchy

| 名称 | 位置或后备存储 | 主要特点 |
|---|---|---|
| Register | SM 片上寄存器文件 | 线程私有，低延迟；寄存器资源有限。 |
| Shared Memory | SM 片上 SRAM | 同一 Thread Block 内线程协作和数据复用；由程序显式管理。 |
| Local Memory | **线程私有的逻辑地址空间**；通常由设备 DRAM 提供后备存储 | 例如无法放入寄存器的线程局部数组、寄存器 Spill；访问可经过 L1/L2 Cache。 |
| Global Memory | 设备全局地址空间；通常由 HBM 提供后备存储 | 容量大，多个线程可访问；访问可由 L1/L2 缓存。 |
| L1 Cache | SM 片上 | 主要服务当前 SM，A100 的 L1 数据缓存与 Shared Memory 使用统一的片上资源体系。 |
| **L2 Cache** | **GPU 芯片上，多个 SM 共享** | 缓存全设备的访存流量；**不是某个 SM 私有**。 |

**订正：** 原回答把 L2 Cache 写成“片上 SM”，需要改为“片上、GPU 范围共享”。Local Memory 名称里的 *Local* 指线程私有的地址空间，不代表一定在 SM 内部，也不意味着每次访问都绕过缓存直达 HBM。

### Q2. Constant Memory 与 Texture Memory

- **Constant Memory**：对一个 Warp 内多个线程读取**相同地址**特别有利。若访问命中 Constant Cache，硬件可以对相同地址执行广播；如果 Warp 访问许多不同地址，访问可能按不同地址序列化。
- **Texture Memory**：不是一块独立的 HBM，主要指通过 Texture Object 等机制进行的纹理读取；适合具有空间局部性的访问，且可支持坐标寻址和特定格式下的硬件插值。

**订正：** 不是“一个线程先拿到、再把数据广播给其他线程”，而是**硬件的 Constant Cache/读取路径将同一地址的数据广播给 Warp 内需要它的线程**。普通 Global Memory 也可能缓存命中和高效合并访存，因此 Texture Memory 不一定更快。

### Q3. Stride 与 Memory Coalescing

对于 `shape=(4, 8)`、`stride=(1, 4)`：

\[
\operatorname{offset}(i,j)=i\times1+j\times4
\]

所以 `x[0,0]`、`x[0,1]`、`x[0,2]` 的**存储偏移分别是 0、4、8 个元素**，相邻访问地址相差 4 个元素。FP32 下相邻访问地址相差 16 Bytes。

**订正：** 原回答没有写出完整的三个偏移。Tensor 不连续不等于 GPU 访问一定低效；是否合并访存，取决于相邻线程实际访问的地址及 Kernel 的线程映射。

### Q4. Effective Bandwidth 与 HBM Bandwidth

\[
BW_{\mathrm{effective}}=\frac{\text{Logical Bytes Read}+\text{Logical Bytes Written}}{\text{Kernel Latency}}
\]

逻辑有效带宽使用**程序语义估算**的数据搬运量；实际 HBM 流量则必须关注 DRAM 接口真实传输的字节数。缓存命中、写入路径、重复访存和额外内存事务等，都会让两者不同。

如果计算结果超过 HBM 标称带宽，首先检查：

1. MiB/MB、GiB/GB、ms/s 的单位换算，读写次数和 FLOPs/Bytes 公式。
2. CUDA Event 的测量范围、预热、重复次数、是否计入实际 Kernel。
3. 数据是否从 L2/L1 Cache 获取，而非全部经过 HBM。
4. 设备型号、运行模式、时钟、共享 GPU 负载及所比较的峰值带宽规格。

**要点：** 逻辑有效带宽不是 Nsight Compute 的 DRAM Throughput。

### Q5. Arithmetic Intensity 与 Roofline

\[
AI=\frac{\mathrm{FLOPs}}{\text{Bytes Transferred}}\quad(\mathrm{FLOP/Byte})
\]

\[
P_{\mathrm{attainable}}\leq \min\left(P_{\mathrm{peak}},\ BW_{\mathrm{peak}}\times AI\right)
\]

\[
AI_{\mathrm{ridge}}=\frac{P_{\mathrm{peak}}}{BW_{\mathrm{peak}}}
\]

- **Ridge Point（拐点）左侧，低 AI：Bandwidth-bound 区域**，Roofline 斜率由带宽决定。
- **Ridge Point 右侧，高 AI：Compute-bound 区域**，上界由计算峰值决定。
- 上面只是**模型给出的主导上界**，不能单凭落在某侧，就断言一个未达到上界的真实 Kernel 已被该资源实际饱和。

**重要订正：** 原回答将左右两边写反了，并且把 `P = AI × BW` 当成完整 Roofline。完整公式要取 `min(compute roof, memory roof)`。AI 的 Bytes 必须明确选择 HBM、L2 等相应内存层级。

### Q6. Roofline 为什么不能直接给出优化方案？

两个 Kernel 的 AI 均为 2 FLOP/Byte、Roofline 上限均为 2 TFLOP/s：A 实测 1.9 TFLOP/s，B 实测 0.5 TFLOP/s。**B 的性能差距更大，更值得优先调查。**

但只有 Roofline 位置不能确定 B 的根因。可能存在访存合并差、内存延迟隐藏不足、可执行 Warp 数不足、指令依赖、寄存器/Shared Memory 限制、指令吞吐不足等情况。需要结合 Nsight Compute 的 Memory Workload、Scheduler Statistics、Warp State Statistics、Occupancy 等指标验证。

**区分：** *接近 Memory Roof* 表示在模型假设下可能接近所选内存带宽上界；*远离 Roof* 只表示有性能差距，并不自动指向某一个优化策略。

## Part B — 计算题（含订正）

### Q7. Copy 有效带宽

`FP32 Tensor shape=(1024,1024)`，读取一次、写入一次，耗时 0.02 ms。

- 单个 Tensor：`1024 × 1024 × 4 = 4,194,304 Bytes = 4 MiB ≈ 4.194 MB`。
- 总逻辑读写量：`2 × 4,194,304 = 8,388,608 Bytes = 8 MiB ≈ 8.389 MB`。
- 时间：`0.02 ms = 0.00002 s`。
- 有效带宽：`8,388,608 / 0.00002 / 1e9 ≈ 419.43 GB/s`。

**订正：** 原答案只有除以 `1e6` 的表达式，没有完整写出时间换算与带宽结果。**MiB = 2²⁰ Bytes，MB = 10⁶ Bytes**。

### Q8. Elementwise Add Arithmetic Intensity

`z = x + y`，FP32，每输出元素：

- 计算量：1 次加法 = **1 FLOP**。
- 逻辑搬运量：读 x 4B + 读 y 4B + 写 z 4B = **12 Bytes**。
- `AI = 1 / 12 ≈ 0.0833 FLOP/Byte`。

低 AI 表示在大规模、数据主要来自 HBM 的假设下，更可能受到**内存带宽**约束；如果在缓存中重复读取，需要改用相应层级的实际流量重新分析。

## Part C — A100 Copy / Add Benchmark

实验采用 CUDA Events、Warmup 和重复测量，统计的是**逻辑有效带宽**，不是硬件计数器测得的 HBM 带宽。

| 单个 Tensor Size | Copy 延迟 (ms) | Copy 有效带宽 (GB/s) | Add 延迟 (ms) | Add 有效带宽 (GB/s) |
|---|---:|---:|---:|---:|
| 4 MiB | 0.018 | 455.111 | 0.017 | 722.824 |
| 16 MiB | 0.034 | 992.970 | 0.049 | 1024.000 |
| 64 MiB | 0.109 | 1236.528 | 0.156 | 1293.474 |
| 256 MiB | 0.396 | 1354.749 | 0.584 | 1379.705 |

> 注意：延迟仅显示三位小数，不能直接用表中**四舍五入后的时间**重算出完全相同的带宽。应使用 Benchmark 中未舍入的原始计时结果计算。

### Q9. 为什么有效带宽随规模增大而提高？

小数据规模时，Kernel Launch/调度等固定成本难以摊薄，且 GPU 可能没有足够并发访存请求充分使用内存系统；规模增大后可以提高有效带宽。但**不能根据这条曲线断言小 Tensor 的瓶颈是 L2 Cache**：小工作集可能更容易缓存命中，同时仍可能受到 Launch 和并行度等限制。需要硬件计数器区分。

### Q10. Copy 与 Add 为什么有效带宽不同？

- **Copy：** 每元素读取 4B，写入 4B，逻辑搬运 **8B**。
- **Add：** 每元素读取两个输入共 8B，写入 4B，逻辑搬运 **12B**，另有一次浮点加法。
- **订正：Add 的逻辑字节数更大，不是 Copy 更大。** 因此 Add 即使耗时更长，按照“字节数 / 耗时”计算的有效带宽也可能更高。
- 两者的实际访存路径、Load/Store 组合、缓存行为、指令开销、并发与资源使用也可能不同。不能只凭有效带宽数值比较哪个 Kernel 更好。

### Q11. 1379.705 GB/s 是否证明 Add 已经 HBM-bound？

**还不能证明。** 256 MiB 测得约 **1.38 TB/s 逻辑有效带宽**，与 A100 PCIe 40GB 约 **1.555 TB/s 的理论 HBM 带宽**较接近，**支持“大规模 Add 可能受 HBM 带宽约束”这一假设**，但并非直接证据。

下一步可使用 **Nsight Compute** 测量 DRAM read/write bytes、DRAM throughput / utilization、L2 hit/traffic、Warp Stall 和 Memory Workload 指标，判断实际 HBM 是否接近饱和，并结合改变数据规模、访问模式的对照实验验证。不要直接用逻辑有效带宽当作 DRAM 实测带宽。

## Part D — 补充：并发度不足、TLP、ILP 与 Warp Stall

这部分来自 Day 3 后续针对 Roofline 差距的讨论，属于进一步 Profiling 的重要概念。

| 概念 | 关注层面 | 简要含义 |
|---|---|---|
| TLP（Thread-Level Parallelism） | 线程 / Warp 之间 | 存在多少独立 Warp 可以被调度，能否通过切换 Warp 隐藏延迟。 |
| ILP（Instruction-Level Parallelism） | 同一个线程 / Warp 的指令 | 是否有多条相互独立的指令链，能够减少对前序指令结果的等待。 |
| Warp Stall | 单个 Warp 的调度状态 | 某个 Warp 暂时没有可发射的下一条指令，例如等待内存、依赖结果或同步。 |
| Eligible Warps | Warp Scheduler | 当前有多少 Warp 具备可发射指令，是判断能否隐藏 Stall 的关键指标之一。 |

**核心关系：** 广义的“可执行独立工作不足”可能来自 **TLP 不足**，也可能来自 **ILP 不足**，或者两者共同作用。Warp Stall 与并发度不足可能同时出现，但**不是同义词**：即使部分 Warp 正在 Stall，Scheduler 仍可能通过其他 Eligible Warps 持续发射指令。

**分析顺序（启发式，不是单指标定论）：**

1. 看 Active Warps / Occupancy：是否有足够 Warp 驻留？
2. 看 Eligible Warps 与 Issue Slot Utilization：有多少就绪指令可以发射？
3. 看 Stall Reasons、DRAM/L2 吞吐、寄存器与 Shared Memory 占用：究竟在等待什么？
4. 提出一个可验证的优化假设，再调整线程映射、数据复用、Block 配置或独立计算链，并重新 Profiling。

**重要限制：** 提高 Occupancy 不保证加速；增加 ILP 也可能增加寄存器压力。真正的瓶颈可能是带宽上限，而不是可执行 Warp 数量。

## Part E — Revised Summary（定稿）

今天学习了 GPU 的内存层次：Register、Shared Memory 位于 SM 片上，L2 Cache 在 GPU 芯片上由多个 SM 共享；Global Memory 与线程私有的 Local Memory 通常以 HBM 为后备存储，但访问可经过缓存。性能分析时要区分程序层面的逻辑读写字节数和实际经过 HBM 的流量。Arithmetic Intensity（AI）是 FLOPs 与所选内存层级数据传输量之比；Roofline 用计算峰值和带宽峰值为不同 AI 的 Kernel 提供性能上界。Ridge Point 左侧是带宽上界主导区域，右侧是计算上界主导区域。Roofline 能揭示性能差距，却不能直接指出是访存效率、TLP/ILP、Warp Stall 还是其他原因。A100 实验中，Copy 与 Add 的逻辑有效带宽总体随数据规模增加而提高，256 MiB 时分别约为 1.35 和 1.38 TB/s，接近 HBM 标称带宽，但要证明 HBM-bound 仍需 Nsight Compute 的真实 DRAM/L2 流量与调度指标。

## Part F — Feedback / Corrections 与验收

### 已掌握

- Constant Memory 与 Texture Memory 的典型访问特征。
- Roofline 可用于发现性能差距，不能凭图直接确认具体根因。
- Elementwise Add 的 AI 为 `1/12 FLOP/Byte`。
- 小规模 GPU 操作的固定开销可能影响有效带宽。

### 本次修正的主要错误

- **Q1：** L2 Cache 是 GPU 级共享缓存，而不是 SM 私有。
- **Q2：** Constant Cache 是硬件广播，不是先由一个线程读完再转发。
- **Q3：** 偏移为 `0, 4, 8` 个元素。
- **Q5：** Ridge 左侧 **Bandwidth-bound**，右侧 **Compute-bound**；完整 Roofline 公式需要 `min`。
- **Q7：** Copy 总流量 **8 MiB**，有效带宽 **419.43 GB/s**。
- **Q10：** Add 每元素 **12B**，Copy 每元素 **8B**。
- **Q11：** 逻辑有效带宽接近理论 HBM 峰值，只是 HBM-bound 的支持性迹象，不是证明。

### 下次间隔复习（无需现在额外做）

1. 不看笔记解释为什么 `stride=(1,4)` 的访问 `x[0,0:3]` 是跨步访问。
2. 不看笔记写出完整 Roofline 公式并指出 Ridge 两侧的含义。
3. 解释为什么 1.38 TB/s 的**逻辑**有效带宽仍需 DRAM 计数器验证。
4. 给出 Warp Stall 与 TLP/ILP 不足的区别，并说明为什么低 Occupancy 不能单独证明瓶颈。

### 当前状态

- [x] Q1–Q11：已完成逐题反馈与修订
- [x] Q7–Q8：单位和计算已核对
- [x] Q9–Q11：已补齐实验分析与证据边界
- [x] Revised Summary：已完成
- [ ] 自己闭卷重述 Q5、Q11，并确认理解
- [ ] Git Commit

```bash
git add day03_notes.md memory_benchmark.py
git commit -m "Day 3: analyze GPU memory bandwidth and roofline"
```

## 参考阅读

- [NVIDIA CUDA Best Practices — Device Memory Spaces](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-spaces)
- [NVIDIA CUDA Best Practices — Effective Bandwidth Calculation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#effective-bandwidth-calculation)
- [NVIDIA Nsight Compute — Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#roofline-charts)
- [NVIDIA Nsight Compute — Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
