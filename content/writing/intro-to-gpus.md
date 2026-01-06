---
title: "Intro to GPUs For the Research Oriented"
date: 2026-01-05T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "systems"]
tags: ["llms", "systems"]
description: "Getting comfortable with the hardware on a quest for more MFU."
summary: "Getting comfortable with the hardware on a quest for more MFU."
ShowToc: true
TocOpen: false
draft: false
---

Over the back half of last year, I've started touching on a lot of what I would call _systems-y_ ML work. That is, a lot of the things I've wanted to do with LLMs have involved going below comfortable levels of abstraction into what the physical hardware is doing. In particular, this showed up when working on increasing Model FLOPs Utilization (MFU) for a [training job I was working on](https://x.com/shncldwll/status/1992312563749806512). I pulled a few knobs available to me from torchland and got to see the number go up. Stuff like changing the batch size, the tensor quantization, pretokenizing my data, they were all available at the top level model code and I saw a significant increase in MFU.

The thing that took me the furthest though, by a wide margin, was swapping out my naive implementation for FlashAttention.

https://x.com/shncldwll/status/1992312563749806512

I've been finding myself bothered by not understanding the _why_ of that. I know on some level that the implementation is more memory efficient, but I don't have a good first principles understanding of why that's true, nor could I write it myself. I had a similar feeling last year using Unsloth for the first time, which let me train a bigger model on longer contexts. There were systems tricks being done that I didn't understand. This felt very limiting, since I was then subject to working on projects where my preferred software stack supported the techniques I was interested in using for research.

I feel LLMs are getting good enough at writing systems code that soon that won't be a real limitation and it will be relatively "cheap" to support a fork of training code that does what you want it to do. At the same time, it's clear to me that if you don't understand what you're asking an LLM to do, it won't be able to perform nearly as well as if you _did_ understand the problem. In that sense, even autonomous coding agents feel like more augmentation than autonomy. If you can jump a metaphorical five feet in a particular area of programming problems, you can get the LLM to jump fifty feet. If you can only jump one foot, well...[^2]

So, it behooves researchers and research engineers in particular to have a deeper understanding of systems ML programming, both for our own ability to engage with the literature and to be a better steward to any coding agents we're working alongside. That's easier said than done, since GPU programming is considered notoriously difficult, black-magick-y kind of work. I've spoken to plenty of big lab researchers who talk about legends of ML like they're regular people who happen to be good at their jobs. They talk about GPU people quite a bit differently:

{{< x id="1968027724913709071" user="khoomeik" >}}

{{< x id="1968140448062746651" user="itsclivetime" >}}

But as Silvanus P. Thompson said, 

> What one fool can do, another can.

In this post, I'll provide a gentle introduction to learning about GPU kernels in the context of writing more efficient ML training jobs, and share a roadmap I've found useful for those looking to do the same. 

I find tackling similar concepts at different levels of abstractions really reinforces concepts, so we'll be moving up and down the stack as required to make the points necessary. By the end, you should understand elements of the physical design of the GPU that informs how you would design an efficient kernel at whatever level of the software stack you choose to do that. 

# Why go GPU mode?

Not all performance engineering is GPU-related. We've discussed in past posts, for example, optimizations aimed at overcoming [network-bandwidth bottlenecks](https://hackbot.dad/writing/data-parallelism-for-the-poor/). In general, you should first strive to know what your bottleneck is before you try to find techniques to tackle it. Briefly, we can break down each bottleneck type in the Pentagram of Performance[^1] as follows:

**Compute**: You're bound by doing GEMM (**Ge**neral **M**atrix **M**ultiply) operations.  This one is hard to do anything about. You have a GPU that does a certain amount of teraflops, and you're bound by the flops you've got to do for your operation. If you buy a more expensive GPU, you can get more TFLOPs.

**Overhead**: The grab bag for stuff that isn't the other stuff on this list. In particular, this might look like eager execution mode in PyTorch. Your GPU is waiting for python to dispatch the next CUDA kernel, and while it's waiting on that it's not doing any matrix multiplication. 

To illustrate this, take a look at this [old blog on CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) which has the following illustration:

{{< figure src="cuda_graphs.png" alt="" caption="Just gotta keep those things busy." >}}

In eager PyTorch world, there's no assumed knowledge of the graph and each kernel will launch separately. There's a latency incurred, but more importantly you don't start queuing the next kernel until you've got the result of the last one back. The GPU has to wait while the CPU prepares more work for it, and sits idle. If you've got the entire graph of operations, it takes a little bit of time to create that graph and dispatch it, but there's very little latency between the kernels. That's an example of reducing overhead. 

**Data-Loading Bandwidth**: Say you have to load data through the cloud, or reading from disk takes a long time because you don't have an SSD or something. Reducing this is an example of why long training jobs have usually tokenized the data before they read it in. If you're streaming samples in (which you have to be) you don't want to incur the additional cost of waiting for them to be properly processed before they can get on the GPU. You want this data in the GPU's DRAM before it's needed for a kernel. 

**Network-Bandwidth**: All distributed training is going to incur some sort of cost. Activations might need to move between nodes, or data parallel nodes may need to run an all-reduce. All of that time being taken up is time your GPUs can't do matrix multiplications. DiLoCo, for example, reduces the frequency with which data parallel workers need to communicate. The [Intellect-1](https://www.primeintellect.ai/blog/intellect-1) report also mentions int 8 quantization being performed on the DiLoCo psuedo-gradients in order to reduce the payload size. This is another example of tackling network-bandwidth bottlenecks, though an extreme one. 

**Memory-Bandwidth**: This is moving data from one place in memory to another. Writing from the host to the GPU? Memory bandwidth. DRAM on your GPU to SRAM? Memory bandwidth. This is the big one that justifies writing custom kernels, so it's worth backing up to explain GPUs a bit so you can better spot these kind of problems.

Briefly, you've probably heard of **operator fusion** or **kernel fusion** associated with memory bandwidth issues. One of the goals of this blog post is for you to understand that more intuitively.

# Intro to GPUs

Most people I read about in the optimization space recommend writing kernels for ML in Triton. That said, I think for pedagogical reasons, you should start with CUDA. 

Triton introduces a lot of abstractions to make your life easier, but I think you appreciate why those abstractions are there and design your kernels differently if you understand them better. Let's take a look at the "Hello World" of GPU programming, the vector addition. 

A **kernel** is just the function you write that is designed to be run on the GPU. They won't return anything. Rather, they're passed pointers to global memory and will mutate that. Literature will refer to "host" and "device" code. The **host** is the CPU (it's the one dispatching work to the GPU!), and the **device** is the GPU (it does the work!). 

```c++
__global__
void vecAddKernel(float* A, float* B, float* C, int n){
	int i = threadIdx.x + blockDim.x * blockIdx.x; //indexing
	if (i < n){
		C[i] = A[i] + B[i];
	}
}
```

The actual calculation isn't interesting, but there are two things that are. 

First, the calculation of the index. Why do we need one? This is because the execution of kernels is concurrent and parallel. One copy of this program is executing _per-thread_. We have no idea which thread is going to execute first, and multiple copies are almost certainly running on different pieces of hardware simultaneously. As the programmer, we want to make sure we're taking care of each element, and the index is the way of finding what a given thread is "responsible" for.

Next is the boundary condition. `i < n` implies that we're going to launch more threads than our arrays have elements, so we want to avoid reading from or writing to memory that would be out of bounds of that. To see why that is, let's look at the "host" function that would launch that kernel.

```c++
void vecAdd(float* A, float* B, float* C, int n){
	float *A_d, *B_d, *C_d;
	int size = n * sizeof(float);
	
	cudaMalloc((void **) &A_d, size);
	cudaMalloc((void **) &B_d, size);
	cudaMalloc((void **) &C_d, size);
	
	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
	
	vecAddKernel<<<ceil(n, 256.0), 256>>>(A_d, B_d, C_d, n);
	
	cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
	
	cudaFree(A_d)
	cudaFree(B_d)
	cudaFree(C_d)
}
```

Note that we're copying arrays from the host to the device. This is copied into the GPU's DRAM, which is what shows up when you query `nvidia-smi`. You do this all the time in PyTorch, though less explicitly, when you call `.to(device)`

Then, after the kernel is run, you need to access the result of your computation and pull it back to the host memory. 

The interesting line is: `vecAddKernel<<<ceil(n, 256.0), 256>>>(A_d, B_d, C_d, n);`, which looks like a templated function call. You can interpret that syntax like this:

`foo_kernel<<<(num blocks, threads per block)>>>`

The threads you're running will be organized into **blocks**. All the blocks and all the threads running within them are known as the **grid**.

The idiom here `ceil(n, 256.0)` is a common one. The number of threads you need is determined by the size of your input data. For `vecAddKernel` we're launching in sets of blocks of size 256. So say $N$ is 20,000. `ceil` is going to round up the result of 20,000 divided by 256 (which is 78.125) , and get 79. That gives us 20,224 threads to work on our input of size 20,000. That's why we need the boundary condition. 

## Dimensions

In the `vecAddKernel`, our grid and block size was one-dimensional. That is, we had 79 blocks and each of them had 256 threads and they were flat.

So when we calculated the index:

`int i = threadIdx.x + blockDim.x * blockIdx.x;` 

`threadIdx.x`: What thread am I in this block?

`blockIdx.x`: What block am I running in? 

`blockDim.x`: How many threads are in each block?

As you might suspect based on the `x` property, you can actually lay out threads in two additional dimensions. The grid and the block can both be separately defined as `dim3` objects.

This is an abstraction that's most useful based on the kind of data you're reading. If you were processing an image, for example, you might choose to organize the blocks as being on a 2D grid to more naturally line up the thread organization with the problem structure. For example, in a matrix-multiply. Assume we are multiplying $M$ by $N$. We could use two dimensional blocks to take care of it. 

```cpp
__global__ void MatrixMulKernel(float *M, float* N, float* P, int Width){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	}
```

For the sake of understanding, assume we did:

```cpp
dim3 grid(2, 2); // 2x2 or four blocks in the grid
dim3 block(2, 2); // 2x2 or four threads per block
MatrixMulKernel<<<grid, block>>>(M, N, P, W)
```

Our grid would look like

{{< figure src="thread_block_index.png" alt="" caption="" >}}

But after we applied the indexing, we would get to our desired global value for the output.

{{< figure src="global_thread_math.png" alt="" caption="" >}}

That's the basic model of CUDA programming. What's missing from this is an understanding of the hardware that can guide you towards better design of kernels. The most fundamental physical-hardware understanding I can think of is _compute intensity_ and the mismatch between memory-bandwidth and FLOPs. 

## Memory Bandwidth

In general, every kernel is going to involve:

1. Reading in one or more pieces of data. 

2. Performing one or more operations on that data.

3. Writing that data back.

GPUs are incredibly specialized pieces of equipment that are really good at matrix multiplication, and their flops are well-advertised. Looking at the H100 spec sheet, we see:

{{< figure src="h100_spec_sheet.png" alt="" caption="" >}}


We've discussed the sparsity feature in [a previous post](https://hackbot.dad/writing/pretraining-at-home/). We're most frequently in training going to be looking at our Peak BF16 Tensor Core TFLOPs, which for this H100 SXM5 is 989.4 TFLOPs. 

That's a lot of FLOPs. In fact, that may be so many FLOPs that we actually struggle to use them all effectively. Because you can't run floating point operations on data you don't _have yet_.

Included in the spec sheet of the H100 (though less advertised) is the 3.35 terabytes/second of global memory bandwidth. That is, not including latency, you can get 3.35 terabytes from DRAM a second. If we assume we're using 16 bit values (two bytes per number), that means we can load in 1.1 trillion numbers in the same time that the GPU can perform 133.9 trillion floating point operations. 

The implication is that we can compute 100x faster than we can actually get access to the data. And how many FLOPs are actually in common operations? 

Consider our previous addition example. We were reading two vectors from memory, and writing one value out. That's $3N$ reads and writes for $N$ FLOPs where $N$ is the size of the input vectors. Even if doing the calculation wasn't one hundred times faster, it's still obvious that we're spending most of our time on reading and writing data. This is a memory-bound operation. 

Ideally, since actually doing the FLOPs is so much faster and we want to maximize them, we would like the number of FLOPs for each memory access to be higher. It turns out that this measure is formalized by the **compute intensity** of a kernel.

The compute intensity is equal to the number of FLOPs over the number of bytes accessed from global memory. If the FLOPs dominate, the compute intensity is higher and it's a "compute-bound" algorithm. If the memory-accesses dominate, meaning the performance of the algorithm is limited by the speed of data transfer, it's memory-bound. 

So based on our understanding of the peak bandwidth of global memory from the H100, and our FLOPs, have the idea that we need to do around $100N$ FLOPs before we start to do enough compute for it to be the limiting factor.

Let's see if we can reproduce the spec sheet numbers of the H100. 

Consider the following function:

```python
def f(x):
	for _ in range(iters):
		x = x * 2
	return x
```

Applied to the following data:

```python
inputs = [torch.randn(2**24, device='cuda', dtype=torch.float16)]
```
This function takes a tensor on the GPU, and doubles it for a certain number of iterations. To do the calculation, we're basically doing a single FLOP per element in the tensor every iteration. This is a contrived function, but it allows us to hold the amount of reads fixed while increasing the FLOPs in a predictable way. 

It's important to note that Triton acting as the `jit` has to fuse the resulting computations - it's not going to load x up for each iteration of the loop or anything like that. 

Let's look at the results:

{{< figure src="roofline.png" alt="" caption="" >}}

For the first few values of `repeat`, you can see we're using next to none of our FLOPs, but nearly all of our bandwidth. Our bandwidth is entirely saturated reading and writing data to and from global memory and it's constraining our runtime. In-particular, you can see that until we get to a value of around 32-64, our runtime is basically flat. Because the bandwidth is the limiting factor, there's essentially no difference in doubling the vector 16 times or 32 times. The calculation is entirely *memory bound* at this level of computational intensity. 

Now remember, the way the kernel is written, increasing values of repeat is changing the numerator in our computational intensity, that is, the FLOPs, while keeping the read/writes of data the same. 

Notably after `64` we start to see our bandwidth utilization fall off a cliff. The compute is now doing enough work that there's a lag before it gives the data back to the memory. We are no longer being held up by our bandwidth, but really just the speed with which we can do the calculations. We have increased that numerator such that we are in a **compute bound** regime. 

This tradeoff from where you're bound based on bandwidth vs being bound on the FLOPs you can do is basically a roofline model. 

**Note**: An eagle-eyed reader would notice this **peak** value of 33.5 TFLOP/s is a lot smaller than what's advertised in the spec sheet for BF16. Best I can tell, PyTorch is upcasting the BF16 to FP32 for the actual computation of the doubling, which for multiply operations is about ~33.5 TF/s on the H100. Since this is just a toy example, I didn't beat it up too much to find the culprit. 

On a basic level, you should now have an intuition for two things:

1. Bandwidth from global memory is a lot slower than FLOPs. 

2. Because of (1), all else equal, it is best if I can **increase** the computational intensity of my kernels while **reducing** the amount of reads and writes to global memory.

Now we can talk about more of the physical design of the GPU. 

## Latency Hiding

So far we've discussed the _bandwidth_ of the memory. That is, the capacity for data that can move through the pipes from global memory to the GPUs SMs.

One thing we haven't touched on yet is _latency_. That is, how long it takes for any data to show up to the GPU once requested. 

Our naive view of the Cuda programming model should suggest that in the operation

```cpp
__global__
void vecAddKernel(float* A, float* B, float* C, int n){
	int i = threadIdx.x + blockDim.x * blockIdx.x; //indexing
	if (i < n){
		C[i] = A[i] + B[i];
	}
}
```

All of our threads in the different blocks in the grid that we launched to do the vector add should be working essentially in lockstep. They'll hit `A[i] + B[i]` and then all need to make requests for that data basically simultaneously, and our GPU will do just about nothing as it waits for that to occur, and we eat the latency. 

It turns out this isn't the case, and we can understand why if we learn a little more about the hardware!

### Warps

**Warps** are yet another collection of threads that occurs within the block level. Currently the collection is grouped in a way that is **hardware-specific**. So right now, a warp has 32 threads in it, but that's not guaranteed to be the case.

GPUs execute their work on **streaming multiprocessors** (SMs), but those processors are physical pieces of hardware and can only operate so many threads at a time. So an SM will get many warps assigned to it. Multiple thread blocks can have warps that share an SM, but we won't break up warps from the same thread block onto different SMs. 

CUDA is said to be an SIMT architecture - single instruction, multiple thread. This is how that happens! At a given time, an SM is running the instructions against a warp of 32 threads. 

One convenient thing about this setup is how it breaks our naive view where we were just waiting around for data to show up. Because we're actually doing the work concurrently within an SM and in parallel across them, we have the ability to hide latency. Nvidia GPUs have the ability to, with zero-cost, switch which warp they're executing. They're able to do this because they have really large registers, allowing them to keep all the threads data available very local to the SM. So if they're running a warp that issues a call to load data, no problem, the scheduler can load a new warp until it hits some kind of high latency instruction, and onto the next warp. 

To get a sense of the numbers, H100s can have 2048 threads in flight per SM, meaning they can have **64 concurrent warps** running on them, meaning it's easy to hide latency by doing different work[^4]. 

One thing that might bother you about this is the way the vector addition kernel is written, it doesn't seem like the latency hiding buys us a lot. Each warp is going to need to read its own data from global memory. The scheduler will recognize this as a high latency, operation, swap to a new warp, which will hit the exact same instructions for a different piece of memory in DRAM, and so on. So why bother? 
## Memory Locality

Many kernels are more complicated than simple element-wise operations. Many of them require sharing information between threads. The way this information is shared results in interacting with a hierarchy of memory, each being successively closer to the SM with a lower latency. This means that as you start to write more complicated kernels, the latency-hiding of the warps can actually be effectively made use of.

Until now, we have only considered DRAM, the "global" memory that all threads in a grid have access to. We've also determine reading from it is very slow.

There is also a much nearer memory within a block, called _shared memory_. All threads in a block have read-write access to this memory. Shared memory is _on chip_ meaning it can be accessed very quickly. So if you're working on an algorithm where your thread blocks can efficiently make use of their intermediate calculations, you can use shared memory to reduce the amount of strain you're putting on global memory bandwidth. 

Consider the following example, a dot product from CUDA By Example:

```c++
__global__ void dot(float* a, float* b, float* c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	
	float temp = 0;
	while (tid < N){
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	
	// set the cache values
	cache[cacheIndex] = temp;
	
	// synchronize threads in this block
	__syncthreads();
	
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	while (i != 0){
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}
```

**Note: This algorithm will not finish the reduction of the dot product, leaving one element per block launched in C. The final reduction is then performed by a simple sum on the CPU.**

Understanding the specifics of the grid-stride loop is less important than seeing the use of the shared memory `cache`. Each thread is writing to its own element in `cache` (note that they only need to know their local index, since the shared memory is block specific). In order to perform the reduction, other threads will need that intermediate data, so it's shared, and is ultimately read from multiple times. You want to limit the amount of times you reach for a specific element from global memory. If you can make it one time, you should, since the cost of going to global memory is so high.

The use of `__syncthreads` just ensures that all threads reach that line before continuing, so all writes are completed before a thread reads from the data again. This is helpful because only collections of _warps_ run at the same time. So if you've got a thread block that's greater than 32, you'll need to make sure all warps have caught up to that position before continuing. 

The other thing that's worth noting is that this is the first algorithm we've observed where the block size choice is actually crucial to the correctness of the code.

```cpp
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	while (i != 0){
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
```

If you use that indexing trick when the `threadsPerBlock` is not a power of 2, you'll skip elements. The takeaway here is that the block and grid design choice as the programmer writing the kernel is not arbitrary. If I got into matrix multiplication tiling here the post would get too long, but it's worth looking at if you want to better grasp the connection between the hardware understanding (reading from global memory is slow and painful) and how that leads to code design decisions (design blocks that can make use of shared memory to make as few of those reads and writes as possible).

Once you start getting into more complex kernels, which are the ones that matter, the lack of real C++ knowledge starts to hurt. 

## Why Triton?

In [Triton](https://triton-lang.org/main/getting-started/tutorials/), instead of writing code from the perspective of a single thread, you've got a program working on an entire block of data. You can then write NumPy-style vectorized operations on it. The Triton compiler will figure out the thread-level implementation.  I assume most ML practitioners reading this, like myself, have a greater familiarity with Python than a systems programming language like C++. As a reminder, though, learn enough CUDA to be dangerous to make sure you understand the basics of the hardware. 

The reasons one might want to use it are similar to any level of abstraction you might introduce into a software stack: it's easier to write, and it's more portable. Hand-tuned CUDA is generally faster, but Triton is generally "fast enough". If you need evidence that Triton is "enough" for most use cases, look no further than torch using it by default to create kernels with `torch.compile`.

## `torch.compile`

When you use `torch.compile`, that's acting as the frontend for the compilation process. TorchDynamo is used to capture the computational graph that defines your function. Then it's passed to TorchInductor, which is the backend compiler that generates your accelerator specific code. For most GPUs you might be using, like NVIDIA, AMD, and Intel, this will generate Triton code.

When learning Triton, it's often useful to see what the compiler will output. Consider the following example. We might naively expect these kernels to be compiled separately.

```python
import torch
import os

# Enable Triton code generation output
os.environ["TORCH_LOGS"] = "output_code"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./my_kernels"
os.environ["TORCH_COMPILE_DEBUG"] = "1"

# Two separate operations (unfused)
def unfused(x):
    x1 = x.cos()
    x2 = x1.cos()
    return x2

# Single chained operation (will fuse)
def fused(x):
    return x.cos().cos()

# Compile both
unfused_compiled = torch.compile(unfused)
fused_compiled = torch.compile(fused)

# Trigger compilation
x = torch.randn(1024, 1024, device="cuda")
_ = unfused_compiled(x)
_ = fused_compiled(x)
```

Both operations are effectively the same. In eager pytorch, the `unfused` one would have 2N reads and 2N writes along with the 2N FLOPs. The `fused` operation would have the same amount of FLOPs, but only N reads and N writes. So is this the sort of thing you break out a custom Triton kernel for? 

`$TORCHINDUCTOR_CACHE_DIR` will create a `my_kernels` directory. In a child directory `nc` you'll see an actual `.py` file that contains the Triton kernel.

```python
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cos_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '8B5C5AF9338C1AAB256A9A579415704A502E73E986488BEC47BA9903BFF4F8C2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cos_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl_math.cos(tmp0)
    tmp2 = tl_math.cos(tmp1)
    tl.store(out_ptr0 + (x0), tmp2, None)
```

Separately, turning on the `TORCH_COMPILE_DEBUG` gives you a bunch of interesting stuff, including Triton code before and after fusion. The code ends up being the same, but it's interesting to look at nonetheless.

In this case, we can see our naive example of illustrating kernel fusion was educational but incorrect. The compiler is able to recognize both our `fused` and `unfused` example have the same computation graph, and fuse them into _literally_ the same function, with one global read and one global write. The compiler is pretty smart, and will fuse 'obvious' things where it sees an advantage!

So if the compiler is smart enough to fuse some operations for better performance, why bother learning to write Triton by hand? What kind of thing is the compiler not smart enough to handle? And since Triton is a higher level language, what can you do from Python that the compiler can't? 
## Online Softmax

Let's take a look at writing Softmax. You've probably called it from torch a hundred times.

$$\text{Softmax}(z_{i}) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

Just looking at the mathematical description, it looks like we're only going to need _one_ piece of global information, which is the sum of the exponents. However, we'll actually want the numerically stable version of softmax. If we take $e^{x_i}$ for large logits, we can easily end up with `inf` in the numerator or denominator, which ruins the calculation.

$Softmax(x) = Softmax(x-c)$, meaning we can subtract an arbitrary constant from each $x_i$ without effecting the final calculation. If we choose $c = max(x)$, then the largest exponent will be $e^0$, which is 1. For the same reason, one term in the denominator will be 1, meaning the sum is at smallest one. The computations then work out cleanly.

So for this to be correct, we actually need _two_ shared pieces of information. We need the max of $x$, and we need the summed exponent. Let's look at the most naive Triton kernel we can think of for that. In this case, we'll assume that the axis of parallelization is that each _program_ handles one row of the input, assuming this is a standard case of softmax being run on logits. 

```python
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    # Each program instance handles one row
    row_idx = tl.program_id(0)
    
    # Pointer to the start of this row
    row_input_ptr = input_ptr + row_idx * input_row_stride
    row_output_ptr = output_ptr + row_idx * output_row_stride
    
    # First pass: find max
    max_val = -float('inf')
    for block_start in tl.range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(row_input_ptr + offsets, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # Second pass: compute sum of exp(x - max)
    sum_exp = 0.0
    for block_start in tl.range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(row_input_ptr + offsets, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(x - max_val), axis=0)
    
    # Third pass: write output
    for block_start in tl.range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(row_input_ptr + offsets, mask=mask, other=-float('inf'))
        output = tl.exp(x - max_val) / sum_exp
        tl.store(row_output_ptr + offsets, output, mask=mask)
```

The most obvious pain point here is that we're calling `tl.load` three times. We're reading in the data first to find the max of each row, second to collect the sum of exponents, and then third to compute the final value and write the output. Can we reduce the number of reads?

It turns out we can, by building up the max and the sum of the exponents at the same time. We can keep the running sum of exponents going, scaling the result based on our current understanding of the max value, like so.


```python
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    # Each program instance handles one row
    row_idx = tl.program_id(0)
    
    # Pointer to the start of this row
    row_input_ptr = input_ptr + row_idx * input_row_stride
    row_output_ptr = output_ptr + row_idx * output_row_stride
    
    # First pass: find max AND sum (online algorithm)
    max_val = -float('inf')
    sum_exp = 0.0
    
    for block_start in tl.range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(row_input_ptr + offsets, mask=mask, other=-float('inf'))
        
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        
        # Rescale previous sum to new max, then add new terms
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
        max_val = new_max
    
    # Second pass: write output
    for block_start in tl.range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(row_input_ptr + offsets, mask=mask, other=-float('inf'))
        output = tl.exp(x - max_val) / sum_exp
        tl.store(row_output_ptr + offsets, output, mask=mask)
```

This is equivalent in terms of output, but lets us get rid of one of our passes. This is _exactly_ the kind of multi-pass reduction the Triton compiler can't handle for us. It requires us to see how we could get to the same result for less. 

We'll bake it off in a really basic way, profiling it on H100 for a fixed column-size that we can reasonably expect a single program to do while varying the amount of rows it has to be performed on. 

{{< figure src="online_softmax_benchmark.png" alt="" caption="It's nice when intuition lines up with profiling." >}}


The online version is in fact around **30%** more efficient than it's three-pass counterpart., despite writing this in high-level Python. I'm a little surprised to see it doing better than PyTorch for some sizes (about 20ms), particularly since I haven't gone through the trouble of auto-tuning it. At any rate, we won't go through the whole tuning process with this because it's not a kernel we intend to place directly into a training and inference job. We'll save that for something meatier like FlashAttention.

What we wanted to show is that our knowledge of the hardware and a desire to reduce global memory reads from CUDA could result in faster implementations in Triton while writing code that is recognizably Python.  

## A Roadmap

This stuff is a lot easier to learn than it used to be. Part of that is just the sheer quantity of resources that didn't exist a few years ago. Below I've listed the resources I've used to get this far. 

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html): Generally explains the motivations of "why" in performance engineering, and lays out the basic concepts you should keep in mind when you're learning.  You should leave this fairly comfortable with the pentagram of compute, memory-bandwidth, overhead, data-loading bandwidth, and network bandwidth, and when each factors in. When looking at any optimization technique, a basic sanity check for whether you're learning anything useful will be whether you can identify what kind of bottleneck you're tackling.  

- [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook): Mostly handles the distributed side of the house. There's a lot here, including some notes on distributed profiling and bits on loading custom kernels. In general, this will get you comfortable with some of the important numbers around tracking training efficiency. If you want your learning to be in the context of _training models_ then this is a reasonable second place to spend your time.

- [Programming Massively Parallel Processors](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311): This is the textbook you'll be told to read when you want to "learn CUDA". It isn't particularly focused on model training, but it's good to understand the basics even if you ultimately plan on using Triton instead. My recommendation is to read the first five chapters and do the exercises. You may feel you've got the gist of the programming after chapter three. However, chapters four and five are more foundational. Chapter four teaches you the architecture of a GPU, and how blocks and warps work with the SMs of the GPU. Chapter five is all about memory bandwidth. You will find it extremely difficult to reason about the efficiency of your kernels without this knowledge. Having bounced off the book after chapter three several times, I found chapters four and five were sort of a skeleton key to a lot of stuff I was reading online. 

- [`srush/GPU-Puzzles`](https://github.com/srush/GPU-Puzzles) is a nice series of GPU puzzles that can be run with numba, which maps Python to CUDA kernels. Many of the exercises are quite easy and will only take you a few minutes, though the harder ones can take a few hours. In general, solving the exercises to be "correct" and then working through what you need to do to get to the optimal amount of global reads/writes will teach you a lot of the mental model of optimization. If you've read chapter five of PMPP, your motivation for doing so will be more clear. 

- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/) the official docs here are strong and go in order. You'll learn how to simulate kernel performance without access to a GPU, and how to profile your performance when you've got it.  I recommend going through each of them, getting the code to run, and then trying to implement it yourself. Try different things and see where they help or hurt performance, taking particular note if where you're reading and writing to memory and how. You can take these in order - once you get to fused attention, well, you probably know as much as me!

- [GPU Mode](https://www.youtube.com/channel/UCJgIbYl6C5no72a0NUAPcTA) has close to 100 lectures covering various topics and a great Discord. If you're interested in performance, there's a community for you here.

- [LeetGPU](https://leetgpu.com/) is like Leetcode but for GPU problems. It has CUDA, Triton, PyTorch, CuTeDSL, Mojo, and recent Jax support. Personally I used it to get exposed to a bunch of different Triton patterns and idioms[^3]

Personally, I don't believe researchers/research engineers can afford to ignore how their hardware works anymore. With LLMs and the amount of communities and projects all based around high performance training and inference, I don't think there's ever been a better time to get started with this kind of work. Good luck!  

[^1]: T-shirts to be released at a later date. 

[^2]: This based on my experience of getting LLMs to write frontend code, where I have tremendously little to offer them besides how I would "like" it to look and feel, rather than specific engineering approaches I prefer for a principled reason. It certainly gets the job done, but i can tell I'm just scratching the surface because I don't know how to ask for what I want. 

[^3]: I also hope to bring more users to the platform. I'm currently ranked 30th in Triton and I should be lower. 

[^4]: This doesn't really matter for the latency conversation, but the SXM5 GPUs have 132 SMs enabled. So ~270k threads going at the same time. 
