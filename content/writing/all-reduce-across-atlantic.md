
---
title: "All Reduce Across the Atlantic: Bandwidth in Decentralized Training "
date: 2025-12-12T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "training", "decentralized"]
tags: ["llms", "training", "research", "decentralized"]
description: "The practical realities of devestatingly high communication cost in training."
summary: "The practical realities of devestatingly high communication cost in training."
ShowToc: true
TocOpen: false
draft: false
---

Since [implementing DiLoCo](https://hackbot.dad/writing/data-parallelism-for-the-poor/) I've been really interested in algorithms for decentralized training. The more [LLMs I train](https://hackbot.dad/writing/pretraining-at-home/) in a "normal" setting, the more interesting to me the decentralized case is. Mostly because seemingly nothing _wants_ to work. So much of the software you can find off the shelf for training is optimized for centralized compute. For good reason, this is the _default_ case. Most people implementing high performance algorithms to train large models are trying to optimize FLOPs, and are doing so in some kind of tightly connected datacenter. It is extremely difficult to keep FLOPs high when you're spending time communicating data. As it is on a single-node, so it is in many.

I implemented DiLoCo, but I tested the algorithm with a cluster of two nodes on [Modal](http://modal.com/). This effectively tests the algorithmic implementation and speaks a little bit to the communication costs of the all-reduce, but fails to capture a real global training scenario.

What can go wrong in a decentralized training scenario? Let's consider just the effects decentralized training has on data parallelism, where each island of compute has access to one model replica. Compute island bandwidth can be variable between nodes. The actual compute can be heterogenous, so the same pipeline and parallelism strategies may not work on each island, requiring a custom setup fit for it, which besides just having different FLOPs may have different memory constraints leading to different minibatch sizes. This would lead certain islands to run ahead of others, so what do you do at the synchronization step? You could wait, but then your faster and probably more expensive compute is hanging waiting for the stragglers. You could try to move ahead, but now the lagging worker is training off of a stale replica and won't catch up anytime soon, and your currently training models are missing information from the dataset you sent to the laggers. Even if the compute was homogenous, there's latency of the islands to each other.   

There's a lot of complicated questions to dig into there, but I'll start by looking at an extremely basic one: Given two nodes on opposite sides of the Atlantic, what's the theoretical bandwidth ceiling, and how close can we get to it for gradient synchronization?

Before we get started, it's worth noting some numbers, bandwidth-wise, for doing data parallel training in a centralized setting. 

### Big boy datacenter numbers

- NVLink for GPUs on the same node: **1,800 GB/s**

- PCIe on the same node: **32 GB/s**

- NVSwitch for nodes on the same rack: **1,800 GB/s**

- InfiniBand on different racks: **50-100 GB/s**

### Indie numbers
- High-end home fiber: 1.25 GB/s

- Typical US Home Internet: 0.026 GB/s


And to understand the size of the data we'd be moving around by default in data parallel training, we can consider the "default" case of passing the weight gradients around in FP32.

10B parameters x 32 bits = 40 GB[^1]. 

So if we were fully saturating the bandwidth, our time to transfer (latency disregarded) would be:

- NVLink: 22ms 
- Infiniband: 800ms
- PCIe 4.0: 1.25 seconds
- High-end home fiber: 32 seconds
- Typical US Home Internet: 26 minutes

We're bandwidth bottlenecked every time our communication time _exceeds_ compute time. If we assume a training step finishes in something like 500ms, we're communication bound by our theoretical max bandwidth as soon as we're going over anything with a smaller pipe than NVLink, and that's before we consider latency. 

Most importantly, we can think of this as primarily effecting [MFU, our main measure of training efficiency](https://hackbot.dad/writing/pretraining-at-home/). Our theoretical FLOPs are defined by the GPU. Any time blocking would essentially be overhead, so our actual achieved FLOPs go down and MFU tanks. 

Let's briefly look at the all-reduce operation we're doing:

### What's All-Reduce for?

Let's look at the dumb version.

```python
import torch.distributed as dist

def sync_gradients(self):
	"""
	Call before optimizer step
	"""
	if not self.sync_grads:
		return
	for param in self.model.parameters():
		if param.grad is not None:
			dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
	torch.cuda.synchronize()
```

When we're training, each replica is working on different data. If they weren't, they would have no novel information to exchange with each other and we'd just be training the same model on each. We're accumulating gradients over minibatches, and building up a local gradient state. Before we run our optimizer step, we want to capture global information from the rest of the replicas about their local gradients. 

All-reduce is a standard distributed computing primitive that lets us take as input some data that is different for each node running the same program (our local gradients) apply some function to them (in this case, averaging) and return the result (the global gradients) back to each program.

This means that the value of `param.grad` for each param now contains the global average of the local gradients, and each program now has the same understanding of the gradient. Now when you take your gradient step, each of them will now be synced, and they will continue to work through their unique batch of data from that new starting point.

So, in that one line, `dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)` we're doing all of that. That's a lot of invisible networking and computation happening in a one liner. What's doing that? Where is the computation occurring? How are the nodes connecting to each other? 

Well, there are several possible libraries that implement those 'collective communication' operations. 

### NCCL

If you remember nothing else about NCCL, remember that it's pronounced "nickel"[^2]. It stands for "NVIDIA Collective Communications Library". The big advantage with NCCL is that it supports GPU to GPU communication and it's optimized for it. That is, when you do all-reduce, the data does not have to be written to CPU. Intra-node, the data can move literally just between GPU memory buffers. Inter-node, you can write from the GPU to the network interface card to the network, the other nodes networking card, and then its GPU. In either case, no CPU.

NCCL itself doesn't handle discovery. `torchrun`/`torch.distributed` handles that instead. When Pytorch runs `init_process_group()` it gets all the processes to check in. Once each rank (process) has joined, they exchange information, and then NCCL establishes direct communication channels to be used when an all-reduce or other collective communication operation is invoked.

Another question - how do those messages actually get passed? It turns out this is dependent on what's available. There's a series of fallbacks implemented to make sure you get the most efficient possible available communication. 

For the same node - If you've got NVLink, you can do GPU-to-GPU interconnect. 

Between nodes, you'll use Infiband with GPUDirect RDMA (Remote Direct Access Memory), so you sort of treat your GPUs as one _big_ GPU for the purposes of communication. 

If you don't have Infiniband, you'll do RDMA over ethernet.

This is all, for decreasing amounts of bandwidth, hyper-efficient and assumes a lossless networking setup. And if all that _fails_ (which it would, in the transcontinental case) you fall back to TCP/IP sockets, which have a lot of algorithms for handling the fact that data is lossy and needs to arrive in specific order. NCCL still technically works, but you'll pretty much never hear about this case because the overhead is high and it's kind of a crazy thing to do. 

### Gloo

Gloo is not an acronym. I think it's pronounced "glue"[^3].

Gloo is, by default, over TCP/IP. The rendezvous mechanism expectation is the same as with NCCL (handled by torchrun) and requires all participating processes to be reachable via a bidirectional network interface.

Gloo is generally known as being intended for CPU-based models. The advice you'll get from [reasonable sources](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=nccl) is that if you're doing GPU training, you should be using NCCL, and if you're doing CPU training, you should be using Gloo. The [Pytorch Docs](https://docs.pytorch.org/docs/stable/distributed.html) get slightly more specific, reading:

>[For GPU hosts with Ethernet interconnect...] - Use NCCL, since it currently provides the best distributed GPU training performance, especially for multiprocess single-node or multi-node distributed training. If you encounter any problem with NCCL, use Gloo as the fallback option. (Note that Gloo currently runs slower than NCCL for GPUs.)

## INTELLECT-1

LLM training is obviously a GPU activity, so just based on these references it seems like NCCL should still be the right call. But, when we read the [INTELLECT-1 Technical Report](https://arxiv.org/pdf/2412.01152), where Prime Intellect trained a 10B parameter model using decentralized compute, we read the following:

>We utilize a VPN because our inter-node collective communication library, **Gloo**, requires all participating processes to be reachable via a private network interface. Additionally, the VPN ensures secure, encrypted communication between nodes.

Intra-node, they use NCCL as you would expect.

{{< x id="1847698511388983372" user="samsja19" >}}


So what's the deal? 

The answer is actually really obvious, and specific to DiLoCo: Prime is legitimately interested in CPU, not GPU, operations during DiLoCo. As I discussed in my [DiLoCo blog](https://hackbot.dad/writing/data-parallelism-for-the-poor/), the algorithm requires essentially two copies of your model.

1. The model being optimized on your local dataset, being run through the "inner optimizer".

2. A replica of the last time you did a global sync, being run through the "outer optimizer."

Storing both of those on GPU would be unnecessarily expensive, and you're really not doing anything with the reference model (not pushing tokens or anything), so you offload it that one to CPU. 

Meanwhile, your inner optimizer step is plowing away, updating weights anywhere from 100 to 500 times, for some tunable parameter $H$. 

Now we get to the interesting part, which is after $H$ steps have been completed. Now you want each replica to share data. You essentially create a pseudo gradient, by looking at the difference between the weights reference model parameters you've got stored in CPU against the parameters you've ended up with after $H$ steps.

The detail I missed in my last blog was this: I assumed you would perform the all-reduce operation on the GPU. That is, you would calculate the gradients on GPU, and then call all-reduce, so you'd use NCCL. [The INTELLECT-1 repo](https://github.com/PrimeIntellect-ai/prime-diloco/blob/main/src/zeroband/diloco.py) doesn't do this, instead it keeps all of the outer model stuff on CPU. The optimizer lives there, and the gradients live there. 

They note this is primarily to avoid VRAM overhead, which I don't entirely understand. It makes sense to not want two copies of the model on GPU, but the all-reduce seems like it could happen on either GPU or CPU without complaint. In-particular when training LLMs, your memory is dominated by the activations moreso than the weights or gradients, so when you're going to block the GPU anyway until the all-reduce is complete and you're ready to start training again, you should have plenty of available VRAM. My current assumption is this is attractive/advantageous because recovering from failure with gloo is easier than handling it on the GPU, but I'm just making that up. It would also be reasonable to me if the sharding from FSDP2 made the VRAM situation more tenuous. If anyone knows, tell me! 

So, after a lot of waffling, it seems that it is as simple as your GPU all-reduce backend should be NCCL, and your CPU all-reduce backend should be Gloo. It's just that you might choose to have some surprising computations on CPU depending on your use-case. 

So, we've got the reported numbers and we understand why doing the all-reduce over CPU is the right choice, making Gloo a natural fit. Let's see what our bandwidth is between American and Europe and whether Gloo can saturate it during an all-reduce. 

## Intercontinental Bandwidth

Using [Prime Intellect](https://www.primeintellect.ai/) I picked up two CPU nodes - one in the United States and one in France. After establishing their Public IPs and connection between them, I wanted to take a look at the bandwidth.

To do that, I used [iperf3](https://github.com/esnet/iperf), a tool for determining the maximum achievable bandwidth on IP networks. There are several tunable parameters, but for our part we were most interested in `-P` the number of parallel streams we run at once.

I started with one stream.

```bash
[  5] local 192.168.0.118 port 39636 connected to 72.46.85.115 port 5201
[ ID] Interval           Transfer     Bitrate         Retr  Cwnd
[  5]   0.00-1.00   sec  6.51 MBytes  54.6 Mbits/sec    0   3.33 MBytes
[  5]   1.00-2.00   sec  12.5 MBytes   105 Mbits/sec  1315    112 KBytes
[  5]   2.00-3.00   sec  11.2 MBytes  94.4 Mbits/sec  1017   1.55 MBytes
[  5]   3.00-4.00   sec  13.8 MBytes   115 Mbits/sec    0   1.64 MBytes
[  5]   4.00-5.00   sec  13.8 MBytes   115 Mbits/sec    0   1.71 MBytes
[  5]   5.00-6.00   sec  16.2 MBytes   136 Mbits/sec    0   1.76 MBytes
[  5]   6.00-7.00   sec  15.0 MBytes   126 Mbits/sec    0   1.79 MBytes
[  5]   7.00-8.00   sec  15.0 MBytes   126 Mbits/sec    0   1.81 MBytes
[  5]   8.00-9.00   sec  16.2 MBytes   136 Mbits/sec    0   1.82 MBytes
[  5]   9.00-10.00  sec  16.2 MBytes   136 Mbits/sec    0   1.83 MBytes
[  5]  10.00-11.00  sec  16.2 MBytes   136 Mbits/sec    0   1.83 MBytes
[  5]  11.00-12.00  sec  15.0 MBytes   126 Mbits/sec    0   1.83 MBytes
[  5]  12.00-13.00  sec  16.2 MBytes   136 Mbits/sec    0   1.83 MBytes
[  5]  13.00-14.00  sec  16.2 MBytes   136 Mbits/sec    0   1.84 MBytes
[  5]  14.00-15.00  sec  16.2 MBytes   136 Mbits/sec    0   1.86 MBytes
[  5]  15.00-16.00  sec  15.0 MBytes   126 Mbits/sec    0   1.89 MBytes
[  5]  16.00-17.00  sec  17.5 MBytes   147 Mbits/sec    0   1.94 MBytes
[  5]  17.00-18.00  sec  17.5 MBytes   147 Mbits/sec    0   2.01 MBytes
[  5]  18.00-19.00  sec  16.2 MBytes   136 Mbits/sec    0   2.09 MBytes
[  5]  19.00-20.00  sec  15.0 MBytes   126 Mbits/sec  226   1.56 MBytes
[  5]  20.00-21.00  sec  11.2 MBytes  94.4 Mbits/sec  204   1.18 MBytes
[  5]  21.00-22.00  sec  10.0 MBytes  83.9 Mbits/sec    0   1.25 MBytes
[  5]  22.00-23.00  sec  11.2 MBytes  94.4 Mbits/sec    0   1.31 MBytes
[  5]  23.00-24.00  sec  12.5 MBytes   105 Mbits/sec    0   1.36 MBytes
[  5]  24.00-25.00  sec  11.2 MBytes  94.4 Mbits/sec    0   1.38 MBytes
[  5]  25.00-26.00  sec  12.5 MBytes   105 Mbits/sec    0   1.40 MBytes
[  5]  26.00-27.00  sec  12.5 MBytes   105 Mbits/sec    0   1.41 MBytes
[  5]  27.00-28.00  sec  12.5 MBytes   105 Mbits/sec    0   1.41 MBytes
[  5]  28.00-29.00  sec  11.2 MBytes  94.4 Mbits/sec    0   1.41 MBytes
[  5]  29.00-30.00  sec  12.5 MBytes   105 Mbits/sec    0   1.41 MBytes
- - - - - - - - - - - - - - - - - - - - - - - - -
[ ID] Interval           Transfer     Bitrate         Retr
[  5]   0.00-30.00  sec   415 MBytes   116 Mbits/sec  2762             sender
[  5]   0.00-30.11  sec   414 MBytes   115 Mbits/sec                  receiver
```

Our bitrate here, for one stream, is `116 Mbits/sec` or `0.0145 GB/s` For a 10GB gradient sync, we're looking at around `689.655` seconds, or around 11.5 minutes for the transfer. Prime Intellect notes that their gradient syncs, which involved more nodes and more data for the ring all-reduce, took "around 1 to 7 minutes depending on the configuration". They were also using tailscale, which they note gave a performance hit. So what gives? Did we just pick really bad nodes?

The [INTELLECT-1 blog](https://www.primeintellect.ai/blog/intellect-1) has an additional detail under "Maximising[^5] bandwidth utilization":

>By sharding our DiLoCo pseudo-gradients in a node, we can maximise network bandwidth utilization by opening multiple connections at the same time when performing the all-reduce. This yielded a transfer speed improvement of 8x on some nodes.

So the idea is basically that instead of calling one all-reduce with a single TCP stream prone to minor failures and retries, we might better be able to saturate our bandwidth by doing multiple smaller all-reduces in parallel. The amount of data being sent doesn't change, but that works better with TCP dynamics over long distances, and the overhead to set up the streams is super trivial in comparison to that travel time.

Let's look at **4 streams**:

```bash
[ ID] Interval           Transfer     Bitrate         Retr
[  5]   0.00-30.00  sec   596 MBytes   167 Mbits/sec  1741             sender
[  5]   0.00-30.11  sec   592 MBytes   165 Mbits/sec                  receiver
[  7]   0.00-30.00  sec   540 MBytes   151 Mbits/sec  2105             sender
[  7]   0.00-30.11  sec   538 MBytes   150 Mbits/sec                  receiver
[  9]   0.00-30.00  sec   377 MBytes   105 Mbits/sec  3048             sender
[  9]   0.00-30.11  sec   375 MBytes   104 Mbits/sec                  receiver
[ 11]   0.00-30.00  sec   477 MBytes   133 Mbits/sec  2011             sender
[ 11]   0.00-30.11  sec   475 MBytes   132 Mbits/sec                  receiver
[SUM]   0.00-30.00  sec  1.94 GBytes   556 Mbits/sec  8905             sender
[SUM]   0.00-30.11  sec  1.93 GBytes   552 Mbits/sec                  receiver
```

Much better! Now at `552 Mbits/sec`, we're looking at `0.069 GB/s`, so a total transfer time of 2.4 minutes.

Now let's look at **8 streams**:

```bash
[ ID] Interval           Transfer     Bitrate         Retr
[  5]   0.00-30.00  sec   628 MBytes   176 Mbits/sec  3062             sender
[  5]   0.00-30.11  sec   627 MBytes   175 Mbits/sec                  receiver
[  7]   0.00-30.00  sec   498 MBytes   139 Mbits/sec  2120             sender
[  7]   0.00-30.11  sec   496 MBytes   138 Mbits/sec                  receiver
[  9]   0.00-30.00  sec   636 MBytes   178 Mbits/sec  2684             sender
[  9]   0.00-30.11  sec   636 MBytes   177 Mbits/sec                  receiver
[ 11]   0.00-30.00  sec   597 MBytes   167 Mbits/sec  2867             sender
[ 11]   0.00-30.11  sec   596 MBytes   166 Mbits/sec                  receiver
[ 13]   0.00-30.00  sec   488 MBytes   136 Mbits/sec  2350             sender
[ 13]   0.00-30.11  sec   487 MBytes   136 Mbits/sec                  receiver
[ 15]   0.00-30.00  sec   422 MBytes   118 Mbits/sec  2930             sender
[ 15]   0.00-30.11  sec   421 MBytes   117 Mbits/sec                  receiver
[ 17]   0.00-30.00  sec   549 MBytes   153 Mbits/sec  2691             sender
[ 17]   0.00-30.11  sec   548 MBytes   153 Mbits/sec                  receiver
[ 19]   0.00-30.00  sec   630 MBytes   176 Mbits/sec  3933             sender
[ 19]   0.00-30.11  sec   628 MBytes   175 Mbits/sec                  receiver
[SUM]   0.00-30.00  sec  4.34 GBytes  1.24 Gbits/sec  22637             sender
[SUM]   0.00-30.11  sec  4.34 GBytes  1.24 Gbits/sec                  receiver
```

This gives us `1.24 Gb/s`, which is `0.155 GB/s` and has a sync time of 1.1 minutes. That `GB/s` number looks a lot closer to the crappy home internet number of `0.026 GB/s`.

You can actually see in the code of the DiLoCo implementation their bucketing strategy for the tensors [here](https://github.com/PrimeIntellect-ai/prime-diloco/blob/main/src/zeroband/diloco.py#L107). Each tensor group is handled separately, which would be ideal for opening those parallel streams and saturating the bandwidth. This implementation looks blocking to me, though, so I'm not sure if they did some magic or not, or whether the code that handles the parallelized calls isn't in the repo. 

Being unfamiliar with Gloo, I also ran an experiment to look at my bandwidth doing all-reduces in torch to see if there was evidence of Gloo opening multiple streams when doing the all-reduce. I used Torchrun over public IP.  

```python
def single_allreduce(tensor: torch.Tensor) -> AllReduceResult:
    """
    Perform a single all-reduce and measure it.
    
    This is pure CPU, matching DiLoCo's outer optimizer setup.
    """
    size_bytes = tensor.numel() * tensor.element_size()
    timestamp = time.time()
    
    try:
        dist.barrier()
        start = time.perf_counter()
        
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        
        duration = time.perf_counter() - start
        
        return AllReduceResult(
            timestamp=timestamp,
            duration_seconds=duration,
            size_bytes=size_bytes,
            success=True
        )
        
    except Exception as e:
        return AllReduceResult(
            timestamp=timestamp,
            duration_seconds=float('inf'),
            size_bytes=size_bytes,
            success=False,
            error=str(e)
        )
```

My mean results for the throughput was `0.01575 GB/s`, which is closest to the bandwidth we observed with a single stream. So by default, it appears Gloo does not parallelize the streams. Mostly this suggests doing decentralized training and achieving fully saturated bandwidth means significantly modifying the operation of your default tooling. It's not going to take care of these kind of edge cases by default. You can see why you would be motiviated to find solutions that let you fully saturate that meager bandwidth though, because that's cutting entire minutes off your sync that is idling your GPUs. Your MFU depends on it!

# Factorio With Tiny Pipes

As a researcher, my goal has always been to get the lowest loss/best test score possible. Getting into the systems perspective on ML is mostly just clarifying to me that the engineering perspective on these systems is straight up playing Factorio.

You're still optimizing, just for a different number. You want to maximize (MFU) and everything from customizing GEMM kernels to optimizing inter-node bandwidth is all in service of keeping GPUs running as fast as possible. Anytime you're paying for those GPUs and they're not doing as many FLOPs as possible, you're burning your most valuable resource. 

{{< figure src="bandwidth_transfer_times.png" alt="" caption="The decentralization tax is pretty big." >}}


Decentralized training is doing all that with both hands tied behind your back. You're basically taking the interconnect bandwidth and squeezing it to nearly nothing, and that additional constraint drives algorithmic improvements. While in a centralized setting you might focus all of your time on keeping GPUs saturated, in a decentralized setting your biggest gains are going to be optimizing transferring as little data as possible between nodes, because that's what's keeping your GPUs from doing work. It also suggests a search for different desirable algorithmic properties - for example, while DiLoCo is good for reducing the amount of times you need to sync during training, it's still blocking when it's time to sync. If you were using heterogenous compute and you expected each replica to process their million tokens in a vastly different time frame, you would suddenly desire algorithms that could comfortably handle transferring that data in an asynchronous way - all the better to keep GPUs going. 


[^1]: Worth noting that the activation gradients _do not_ need to be communicated between model replicas, we're only really interested in the _weight_ gradients. 

[^2]: If you say en cee cee el you will be relentlessly bullied.

[^3]:  I don't know how else you'd pronounce it. I don't think you're beat up for saying this one wrong, though.

[^5]: Their work is cool, so we forgive them the use of the S in maximizing.
