# Day 1 — PyTorch Tensor Layout

## 1. Shape 和 Stride 分别表示什么？对于 shape=(2,3,4) 的连续 Tensor，为什么 stride=(12,4,1)？

shape代表形状，stride代表步幅，因为跨越第一个维度需要跨3*4个元素，第二个维度需要跨4个元素所以为（12，4，1）

## 2. 为什么 transpose() 通常不复制底层数据，但可能影响后续 GPU Kernel 的访存效率？

以为transpose之后通常 不连续导致访存效率降低进而影响GPU Kernel
修正：transpose() 通常只交换 Shape 和 Stride 等元数据，不复制数据。但它改变了逻辑维度与物理存储地址的映射关系。如果后续 GPU Kernel 的线程访问模式与数据布局不匹配，可能产生跨步访存，降低 Memory Coalescing 效率。

## 3. view() 和 reshape() 有什么区别？contiguous() 在什么情况下需要复制？

view()如果不能share storage会报错而reshape会进行内存分配等于进行一次contiguous，内存不连续的情况下会进行复制
修正：
reshape() 并不是每次都会复制数据，也不能简单等同于 contiguous()。
操作	是否复制数据	主要作用
view()	不复制	在 stride 兼容时创建新的 Tensor 视图
reshape()	可能复制	尽量返回 View，否则复制数据
contiguous()	可能复制	确保得到指定内存格式下的连续布局

## 4. 一个 shape=(1024,1024) 的 Tensor，使用 FP32 和 BF16 分别占多少 MiB？

4，2

## 5. 为什么理论上 4 GiB 的 KV Cache，不代表 vLLM 运行时 GPU 总显存只占 4 GiB？

因为vllm运行时按block进行分配会有碎片而且还有模型本身的权重以及运行时产生的中间变量，同时也和transformer架构有关