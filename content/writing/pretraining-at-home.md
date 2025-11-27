---
title: "Pretraining at home: 20B tokens from 222 hours to 12"
date: 2025-11-23T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "training"]
tags: ["llms", "training", "research"]
description: "Optimizing training a Llama 3.2 1B model so we can pretrain in a day without going broke."
summary: "Optimizing training a Llama 3.2 1B model so we can pretrain in a day without going broke."
ShowToc: true
TocOpen: false
draft: false
---

Recently [I implemented DiLoCo](https://hackbot.dad/writing/data-parallelism-for-the-poor/) as part of a class on [distributed training](https://maven.com/walk-with-code/scratch-to-scale). The implementation helped me understand data parallelism a lot better. That said, reflecting on my experience over the last month or so, I felt I was leaving a lot on the table. While I trained on a small dataset - enough to verify that DiLoCo was implemented correctly - I hadn't actually _done_ pretraining. I wasn't looking at loss curves on a test set, or running any particular evals to look at the quality of the model I trained. I was just looking at the loss go down and seeing how fast data moved around.

I had internalized that pretraining was essentially a waste of time. Plenty of labs do it, they release great models all the time, and it's much cheaper to post-train those resulting models. That makes me sound lazy. The more reasonable answer is that pretraining experiments are _fiscally irresponsible_. Training an 8B or 32B model to a point where it's "chinchilla optimal"[^1] is expensive. To get a sense for how expensive, we can look at the training time calculator [here](https://huggingface.co/spaces/scratchtoscale/training-time-calculator).

Let's say we want to train an 8B parameter model. Twenty tokens for each parameter in the model leaves us with a desired 160 billion tokens. We'll assume we're competent enough to get to 50% MFU. That means we'd be training for 22 days. At the current market rate for cloud H100s on [Lambda](https://lambda.ai/), paid by the hour, we're looking at 24 dollars an hour. That means out of pocket, the pretraining of that model to get to the _minimum compute-optimal_ amount of data is *$12,672*. For _one run_. Before we talk about storage costs. 

However, there's been a lot of interesting work on "small" language models recently. Take Karpathy's recent [nanochat](https://github.com/karpathy/nanochat), working on training to get the best model possible for around ~$800. There's a certain attraction to this kind of work from an educational perspective. Just understanding every part of the process in miniature is cool. Also, the model's yours - you can do what you want with it. I'm interested in task-specific local models. My ideal model could run on an edge-device and make 200 tool calls in a row and basically would have to look up everything it wanted to know about the world because it isn't spending 100B parameters trying to memorize frozen knowledge irrelevant to its task. 

There's another attraction altogether for those of us used to "old-fashioned" deep learning work, where a significant amount of time was spent on the modeling itself. I've found that _architectural decisions_ of models have started to flutter out of my brain. This new model uses MoE - this one's got a different attention implementation - this ones got RoPE, etc. Reading the papers released with these models, you get a sense of what's "in", and you can even speak to it, but without having implemented it yourself and trained models with it, there's a certain textbook[^2] feel to the knowledge. I find I feel less like a machine learning engineer understanding the model design, and more like a mix of a zoologist and cultural anthropologist. I can see what way the fields moving and how the collected adaptions in the resulting environment have made stronger models. They're just dead facts. 

Accepting that certain things only appear at scale and I'm unlikely to have tens of thousands of dollars sitting around, I want that modeling intuition back.  Let's start basic and say we want to train a 1B parameter dense model to knock the rust off.

Our goals are:

1. Writing a training loop that works
2. Getting a decent MFU
3. Low touch configuration and good experiment tracking

In particular, we would like to be able to run multiple experiments _a day_. So our total wall-clock time-to-train must be under 12 hours.

## Model
I wanted to be simple and straightforward and start with just a "regular" dense model. I ended up choosing the architecture/tokenizer for [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B), for no other reason than I mentally associate it with "normality" for dense models. We'll be starting from freshly initialized weights. 

## Compute
We'll be using [Modal](https://modal.com/) for these experiments. I've found their SDK extremely easy to use which keeps my iteration speed high. I also love that I can just submit a job and know that when it's done, the compute will spin down. I sleep easier knowing I'm not burning credits. They also have free storage until 2026, so I'm not worrying about storage costs for at least a month and a half[^3]. 
## Data
For a 1B parameter model, we'd like to have twenty billion training tokens (plus some extras for a validation and test set). This is our first non-trivial endeavor. 

[FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) is a great pretraining dataset. It's also really, really large. At 44TB of diskspace and 15 trillion tokens, it's overkill for what we want. We'd really like a subset of 20B tokens to reach the 20 tokens per parameter rule-of-thumb for chinchila-optimality. This question of _what_ subset of 20B tokens is, I suspect, a really important and interesting one, but we're mostly going to sidestep it for the moment until we accomplish our initial three objectives. A future post will cover looking at the data and determining how to validate the quality and relevance of those 20B tokens. 

I know I want _high quality_ tokens. The first subset that seemed reasonable is [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), which is a subset of fineweb curated to have highly educational data. Unfortunately, it's about 65 times too large for us at 1.3 trillion tokens. 

There are many random subsets built out of the dataset. The one that's closest to the size we're interested in is [100BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer/sample-100BT), a measly five times what we're interested in.

While browsing the data on HuggingFace suggests that there's no particular order to this dataset, I'm naturally suspicious and wanted to shuffle it. However, we're not going to download all 97.3M documents to shuffle a sample. HuggingFace allows you to stream samples in. It also provides the ability to shuffle. This provided me enough confidence I was getting _random_ samples from the 100BT subset.

Now I wanted to make sure I got the correct token count. 

First I did it the dumbest way possible and wrote a function that took in the name of the dataset, the tokenizer, and the goal number of tokens. Each sample would be processed sequentially, tokenized, and add up to a specific token count.

For the Llama 1B tokenizer looking for 20,000,000,000 tokens, this was going to take about 12 hours. That's not super surprising because I wasn't batching the tokenization, so the process was fairly laborious.

I decided it would be smarter to get a sense of the number of tokens provided by the average document. The function `get_avg_token_count_of_document` [here](https://github.com/SJCaldwell/nanoPT/blob/main/scripts/01_create_data_volume.py) let me tokenize a sample of 100,000 documents to get a sense of the average and median number of documents in my dataset. Running it I found I got an average token count of 999.32, and a median token count of 616. 

I could now assume each document is going to give me about 999 tokens, which gave me a goal document count of about 20 million. I added another 25% buffer to account for the variance between documents, which gave me a goal of 24M documents. I also chose to shoot for validation and test token counts of 100,000,000 a piece.

## Model Implementation

I kept my first implementation pretty vanilla. You can see the original version [here](https://github.com/SJCaldwell/nanoPT/commit/9eeac6b1038efac56275e3a0a2d8513e5ce1e737). I didn't do any optimizations to make it memory efficient, but it ran. In my heart, I knew this wouldn't be the final version that would get me to a complete experiment - I wrote it with naive attention, after all.

I'm not interested in spending whole heaps of dollars, so I went ahead and launched the job on a single H100. I shot for a sequence length of 4096 and a minibatch size of 16, used gradient accumulation so I could hit my target of one million tokens per batch, and hit an OOM error. I got the same error for 8. And 4. Eventually I realized it was only going to run with a minibatch of 1 (for now). 

Where did those OOM errors hit?

```python
# apply rotary position embedding

cos, sin = self.rotary_emb(value_states, seq_length)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
# repeat k/v heads for GQA
key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

# right here
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
```

Calculating `attn_weights`, obviously. That's a big matrix. On the bright side, the loss goes down. 

{{<figure src="initial_training_run.png" alt="" caption="Only have to wait a week and a half for this bad boy to run.">}}

I should note here that the `val_loss` was calculated off of a very small part of my initial validation set. I like getting my loss fairly frequently, and was plotting it every full batch of one million tokens. Because of my minibatch size of one required by the current attention implementation, it was just totally dominating my training time. I decided to replace it with a fixed number of samples - in this case 100, which represents a fraction of a percentage of my 125,125 validation documents. If I was GPU richer, I'd love to set up a job system that would take my model checkpoint, toss it to object storage, and run it against evals without interrupting my training job and posting the results asynchronously as training went. [Ray seems to support this out of the box](https://docs.ray.io/en/latest/train/user-guides/asynchronous-validation.html).

For now, our focus is on reducing time-to-train and fully utilizing the GPUs we're paying for, so subsets of subsets it is.
## Calculating MFU: How much GPU are we wasting?

Looking at the current state of the code, there's a lot of optimizations I can think of that would make the run finish faster. The obvious ones that come to mind:

1. Pretokenizing the dataset to reduce the amount of CPU overhead between batches
2. Moving to BF16 from FP32.
3. Using FlashAttention so I can fit more samples in a minibatch
4. Data parallelism over 8 GPUs gives us a larger effective global batch size.
5. Fusing specific operations or using `torch.compile`.

What I have less of a sense for is how much each of these optimizations actually helps, mostly because I don't spend a lot of time in the torch debugger improving training jobs - we'll get to that. 

Before that, though, there's a metric we haven't calculated yet - Model FLOPs Utilization or MFU. Given a particular piece of hardware with a published spec for its maximum throughput, what percentage of that are we achieving? This can be read as a percentage, essentially your observed throughput over the theoretical peak throughput.

Word on the street is that 50% MFU would be considered _pretty good_. With all our current inefficiencies, we're lower than that. Let's talk about how it's calculated. 

First, we need to know what we're actually being promised at the hardware level. We can find that from a [NVIDIA datasheet](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306).

{{< figure src="nvidia_h100.png" alt="" caption="That's a lot of numbers" >}}

First question: which of these columns matters to us? Looking at Modal's website, we find:

>All H100 GPUs on the Modal platform are of the SXM variant, as can be verified by examining the power draw in the dashboard or with `nvidia-smi`.

Cool. Those numbers are higher, so I like that. It does imply by the defensive tone that many other providers would attempt to fool me by randomly assigning me one or the other and charging me the same price for them. Oh well, probably nothing! Onto the rows: 

You may, like an absolute fool, look at this and with a straight face say to me: "Shane, this is easy to read. This tells us the TF32 Tensor Core, which our model is currently using, gets us 989 TFLOps.", to which I would say, "Hold on there, pal. There's an asterisk."

That asterisk suggests these numbers are *with sparsity*. This leads us to two questions: is sparsity a good thing or bad thing for TFLOP performance, and does our training job count as a sparse or dense job? 

Using my nigh undefeated understanding of human incentives, I infer that sparsity _must_ be the higher number, or that wouldn't be in a spec sheet that got past marketing. Some quick googling confirms this, sparse is faster. Under some specific circumstances - that is, when two out of every four contiguous values is zero, sparse tensor cores skip the zero-value calculations, and that halves the number of operations done and makes the effective TFLOPs twice as high. 

Sounds great. Does that have anything to do with our training? My similarly undefeated understanding of model architecture suggests that there is _no way_ standard LLM training would conform to this 2:4 ratio. Our matrices are not sparse, and when they _are_ sparse, that sparsity is not structured in such a way to take advantage of this. Some specific pruning during inference might be - if you're willing to take some accuracy hits - but not training[^4].

So, these values are actually _2x higher_ than what we would expect to find. That is, TF32 would be 494 TFLOPs. For BF16 (where we're going) it would be 989.5 TFLOPs. I confirmed this by finding the [technical architecture doc](https://www.techpowerup.com/gpu-specs/docs/nvidia-gh100-architecture.pdf), where the dense/sparse split is written out explicitly on page 20.

{{< figure src="actual_spec_sheet.png" alt="" caption="Pro tip: If you find a table with uglier fonts, it's more likely to be accurate.">}}

Now you too can read the basics of NVIDIA specsheets. It won't make your training faster, but at least you know what you're paying for. It also gives us the _denominator_ for MFU.

Now let's tackle the numerator. We want to know what percentage of our theoretical peak we're achieving. The easiest way to calculate that is to know how many FLOPs are processed for a single token, and then how many tokens you're processing.

To calculate the model FLOPs per token during training, the rule of thumb is 6 times the number of parameters in your model. We can break that into the forward and backward passes:

For the forward pass: let's assume the general matrix multiply (GEMM) with the feed forward matrices dominates the transformer's computation (it does). During each matrix multiply, you're looking at two floating point operations - one multiplication per input dimension, and one add to accumulate them. This is 2 FLOPs per parameter. During the backward pass, you have more computation to do - first computing gradients with respect to activations (backprop) and then computing gradients with respect to weights (for the optimizer step). Each of these costs roughly the same as the forward pass. So $2n$ for forward, $4n$ for backward, for a total of six TFLOPs per token processed. 

Finally, we just need to know how many tokens we saw. That can be more or less complicated depending on how your sequences are designed. We'll assume here every sample is padded to be length 4096, or is a full-sized sample.

I've got an example you can check out [here](https://github.com/SJCaldwell/nanoPT/blob/main/nanopt/profiling/track_mfu.py). Nothing fancy. Basically you define your number of tokens processed for step, and call an `update` function every time you do the forwards/backwards. In this case the step will refer to minibatch steps/sequence length. 

Then when it's time to check your MFU, you're just looking at the number of tokens you processed in your minibatch, multiplied by the TFLOPs you must have done to take the step, divided by the theoretical peak you got from the specsheet. In this case, I started at an MFU of 15%. 40% would be pretty good, 50% would make me very happy, so there's room to grow there. Since calculating the MFU is done with several approximations, it's very cheap, so we can just keep it in our training loop without causing problems. 

## Turning on the Profiler

We'd also benefit from information from the torch profiler, which essentially provides timing and percentage GPU utilization for everything we want to do. 

The profiler is implemented as a context manager. Last time I profiled pytorch was back in my CV days probably five years ago, and I usually did it on random branches off of main or in notebooks to check my math. I really only used it for inference. It just seemed really heavy to add to the training code itself. Since then, I've learned a little more about context managers in python. In-particular, `contextlib.nullcontext()`. This lets you use a conditional to setup your context manager. You can use the torch profiler when you want to, or this no-op otherwise, meaning you can easily flip the profiler on and off without a performance penalty. Great!

```python
if config.enable_profiling and global_rank == 0:
	profiler = torch.profiler.profile(
		activities=[
			torch.profiler.ProfilerActivity.CPU,
			torch.profiler.ProfilerActivity.CUDA,
		],
		schedule=torch.profiler.schedule(
			wait=config.profiling_wait_steps,
			warmup=5,
			active=config.profiling_active_steps,
			repeat=1,
		),
		on_trace_ready=torch.profiler.tensorboard_trace_handler(config.profiling_dir),
		record_shapes=True,
		profile_memory=True,
		with_stack=True,
	)
	profiler_context = profiler
else:
	profiler_context = contextlib.nullcontext()
```

I configured ten wait steps and five warmup steps with twenty steps for actively profiling. I figured at that point we'd be well into training and the GPU would be warmed up. 

What you get out is a `pt.trace.json` profile. It's very information dense. You can check it out right from Chrome using `chrome://tracing`, and it looks like this. 

{{< figure src="chrome_trace_viewer.png" alt="" caption="I don't know what any of this is, and I'm scared." >}}

That's a bit intimidating for me. Also, it doesn't give me a big, obvious number to make smaller, just a lot of little ones. 

What I actually wanted, it turned out, was tensorboard. It has a plugin that lets you view the torch profiler traces. You can install tensorboard and the plugin like:

```bash
uv add --dev tensorboard
uv add --dev torch-tb-profiler
```

Then you can see this much less intimidating and much clearer visualization.

{{< figure src="torch_profiler_start.png" alt="" caption="Make big number go down? That I can do.">}}

_Now_ we're talking. I have very simple numbers I would like to make go down. For example, we can see here that 15.5% percent of the profiled time was CPU overhead. We would like that number to vanish nearly to 0. Each time we make a change to our training setup, we'll see how it effects the MFU and how it effects that CPU overhead figure, and optimizing for those two numbers should get us where we're really looking to go: minimum wall clock time for our training. 

## TLDR: Starting Numbers
So, to summarize, with our naive approach we landed at **15% MFU**, **15.5% CPU overhead** during profiling, and an estimated train time (via [calculator](https://huggingface.co/spaces/scratchtoscale/training-time-calculator)) of 222.2 hours with a single H100.

## Single GPU Optimization
Let's go through them one by one. For each, we'll track the MFU, GPU memory utilization, and total time-to-train as predicted by the training time calculator. 

### BF16

The lowest touch start is BF16. This should reduce the size of the matrices we're multiplying, allowing us to get through them faster. From MFU's perspective, it will also _increase_ the peak theoretical TFLOPs as well. So we may expect this number to not move at all or go down, even. However, that should open us up some memory to play with to increase our batch size, which _will_ help our TFLOPs.

While we were in FP32, our memory utilization looked like 97.52% utilization. We'll change `dtype` to a parameter of our training job, swap it to bf16 when putting the model on device and let it rip.

This is basically a no code change.

```python
model.to(device, dtype=dtype)
```

Running it, our GPU memory starts to hover at around 78%-80%. MFU actually goes up by quite a bit to 40%. This is a little surprising. My best bet is that my minibatch of 1 was so close to the maximum amount the GPU could handle that I was decreasing the efficiency of interleaving writing data to the GPU and processing it. I'm kind of making that up. In the future when I'm a FLOPhead maybe that will make more sense to me. We'll take it, though. 

Total time to train: 83.8 hours. 

### Flash Attention 2
Our memory usage is a little lower, but we've still got the massive bottleneck that is naive attention, which we should work through.

I decided to go with `torch.nn.functional.scaled_dot_product_attention` because it's built right into modern versions of pytorch, and uses flash attention. 

MFU went to 55%, GPU memory usage 25%.

Total time to train: 60.6 hours. 

### Batch Size
With my new available memory, I tried batch sizes 16 and 8, but those still failed. 4 worked a treat, though, and was stable for several hours. 

MFU (on single GPU, mind you) 85%, GPU memory usage 25%. 

Total time to train: 39.2 hours.

Functionally, all we've done here is swap out a naive attention implementation for Flash Attention and played around with batch size, and we've cut our experiment time by almost two days.

### Parallelizing

There's a lot more I could do. Flash Attention 3 and torch.compile seem most obvious, and pre-tokenizing my dataset would also give me some benefits. But the biggest thing holding us back is parallelization.

For small models that fit on a single card, we can do distributed training relatively easily. In distributed data parallel training, you place a copy of the model on each GPU. Each GPU gets different data. Everything plays out just about the same, with each GPU doing its own gradient accumulation. Just before the optimizer takes its step, you do an all-reduce on your gradients, averaging the gradients of each worker. Then, when you take your step, each machine will end up with the same copy of the model and get to work on the next data.

If this were cost-free, it would provide us a linear speedup. If our 85% MFU held over 8 GPUs, we could train in less than five and a half hours. However, that GPU communication to average the gradients is pretty expensive, and the time we spend doing that average is time we're not processing any tokens. Our theoretical throughput, however, _will_ rise linearly. So we can expect it to drop somewhat.

The first thing you're going to want to do is use `torchrun`.

```python
def ddp_setup() -> None:
# check if nccl is available
	dist.init_process_group(backend="nccl")
	torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
```

With `torchrun` to run your job. Something like.

```python
from torch.distributed.run import parse_args, run
    args = [
        f"--nproc-per-node={multi_node_gpus}",
        "-m", "nanopt.main",
        config_path
    ]
    run(parse_args(args))
```

This combination of incantations is going to give you access to a few environmental variables.

```python
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
global_rank = int(os.environ["RANK"])
```

Local rank is the rank of the GPU on the device. World size is how many GPUs there are, period. Global rank lets you know what GPU you are on a zero indexed list of all the GPUs, particularly if you're running on a cluster.

While in a [previous blog post](https://hackbot.dad/writing/data-parallelism-for-the-poor/), I implemented DDP from scratch, we're going for speed this time, which means making use of the tools pytorch makes available. In this case,

```python
model = LlamaForCausalLM(LlamaConfig())
model.to(device, dtype=dtype)
model = torch.nn.parallel.DistributedDataParallel(
	model,
	device_ids=[local_rank],
	output_device=local_rank,
)
```

It would be tedious to go over each and every change you need to make for data parallelization, so I'll just provide a few tips based on footguns I ran into.

1. Whenever you're going to log something, check whether you're global rank 0. If you're going to save the state of your model, check that you're global rank 0. If you're printing something because you want to see it later, global rank 0. There's no need to waste computation or storage by repeating that on every GPU. 
2. `DistributedDataParallel` is wrapping your model. The methods you would usually call on your model may be another layer deeper. The easiest way to get around this is to throw a `model.module if hasattr(model, 'module') else model` at it. This shows up when you're checking your state dicts to log the model and that sort of thing. Forward pass still works normally.  
3. MFU tracking needs to take into account your world size. Whatever the theoretical peak is on one GPU, your theoretical peak is now linearly scaled by your number of GPUs (assuming homogeneity). I briefly was getting readouts of 120% MFU. 
4. Your batches are larger, so I'd recommend scaling your gradients. Can't hurt.

With that, I scaled this job up to 8 GPUs and let it rip.
## Final Time-To-Train

Our final MFU on a single node with eight H100s was 40%. The training time calculator shows that as taking about eleven hours to train. Compared to the 222 hours we started with, that's pretty good!

{{< figure src="full_training_run.png" alt="" caption="Not bad.">}}

It's hard to finish this blog post, because there's so much more I know I could do. Pre-tokenize the dataset, play with CUDA buffers, call `torch.compile` while we warmed up, write a kernel in Triton, figure out what 'flex attention' is. Optimizing training jobs is a job in itself, and one I have slightly more appreciation for. I expect I'll come back to all of the above, but ultimately these optimizations were in service of training small models I want to exist. And for that, what I really need to get into is _data_.

If you want to look at the code, you can check it out [here](https://github.com/SJCaldwell/nanoPT).

Until next time. 

[^1]: It is also the case that most models are trained beyond chinchilla optimality and continue to see stronger performance, so the calculations that follow can be considered a "minimum non-wasteful bar to clear". Consider LLama 3 8B being trained on _15 trillion_ tokens. 

[^2]: Deragatory.

[^3]: I am confident this story ends with me waking up some day in February and realizing I forgot to delete the volumes, but that's for another day.

[^4]: I argued with Gemini and ChatGPT about this for about an hour. ChatGPT told me with a straight face that despite the asterisk, that was just an in-group joke that trips up newbies all the time and that the TFLOPs reported in the above table were dense. After I found a much longer 100 page PDF that showed the dense/sparse values explicitly, it relented. I propose an exciting new benchmark would be testing LLMs against NVIDIA's marketing.