---
title: "DiLoCo: Data Parallelism for the Datacenter Poor"
date: 2025-10-03T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "rl", "distributed"]
tags: ["llms", "training", "distributed"]
description: "Distributed training sans datacenter."
summary: "Distributed training sans datacenter."
ShowToc: true
TocOpen: false
draft: false
---

I'm a big believer in private models. I always have been. The term "local" model still strikes me as strange, because it was previously the default. We'd just call them _models_. If we had to refer to a third party hosted model, we'd just say "the default google model" or whatever, and that was generally derogatory.

Part of this is just when I started in the field. I became an ML engineer in 2018, and finetuning or training models from scratch was just _what you did_. There were a few API-based models you could call, but outside of sentiment analysis they were nearly all uniformly very bad. My friends and I mostly used them as evidence to our boss that we needed to invest more budget in training[^1]. 

Another part of this is a sort of functional professional paranoia. If I put out a product, I'm in some sense responsible for its reliability. If you're an API wrapper, there's very little guarantees you can make. Will my performance be consistent? Will the model be up? Will I wake up one day to find the model is deprecated? I have no idea, man, I just call the API and hope for the best. There are benefits to this, sure, your product can just _get better_ with no effort on your part, but it can also just _get worse_ or _stop existing_. 

Finally, and most important to me if I'm being honest, it's a professional pride thing. I'm a scientist and an engineer, and for the largest part of my career my responsibility has been making models. You want some weights that do a thing, I go through the effort of collecting data, training a model, iterating on it, serving it, improving it. It feels really good to do. You end up being SOTA at some insanely domain-specific stuff. For several years I worked primarily on object detection for household objects for a moving company. The amount of mental energy I spent on data augmentation for occlusion would boggle your mind. To go through that effort and see it work gives you an insane amount of dopamine. Calling an API, frankly, doesn't hit the same.

So, to reduce the probability of calling APIs for the rest of my life, it's time to hit the books.

# Hitting the Books

The goal is to competently train competitively performant LLMs. I've done quite a bit of finetuning of smaller models. Take an A100 and a small Qwen or Llama, finetune it for some particular task, or do a little GRPO. But to train something larger (>30B) and on longer context lengths (128k), I need some skillsets I don't have. In-particular, distributed training. 

Over the past eight years I've been in the field, multi-gpu and multi-node training has gone from a nice-to-have to necessity. Working in computer vision, I might be finetuning a YOLOv8 model that had, on the upper end, around 50M parameters. Running out of GPU memory wasn't a significant concern of mine. When I had access to multiple GPUs, my primary dimension of parallelization was running different training jobs on each GPU in order to speed up hyper-parameter sweep. It's likely I could have trained slightly faster if I had invested time in becoming comfortable with the torch profiler, but it just wasn't a showstopper. The compute was relatively cheap. In general, I found it was much more productive to spend time looking at the data, collecting more data, and introducing new data augmentations. I only looked into serious performance improvements for models when I was putting them on mobile, and that could mostly be done with some kernel fusion and futzing with `torch.compile`. High performance distributed training just wasn't a muscle I stretched very often.

Necessity, however, is the mother of getting-your-act-together. 

I bounced off [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) a few times. Mostly just because I was reading it and not applying it[^2]. The concepts are all there, the exercises are more choose-your-own-adventure. The correct course of action was to just pick something and work on it, but when you're [busy](https://dreadnode.io/) it helps if you've got a little bit of handholding and lot of forcing function. Thankfully, I got the forcing function I was looking for with [Scratch to Scale](https://maven.com/walk-with-code/scratch-to-scale) from [Zach Mueller](https://x.com/TheZachMueller), a class on taking the many distributed techniques necessary for making large model training practical and making you implement them. In addition, he had a totally insane set of lecturers from Unsloth, Prime Intellect, Ray, Huggingface, etc, each of whom is world-class at their particular part of the stack. 

I'm not an online class person. I hate my schedule being dictated by someone else. I've got a job for that! But the syllabus looked like exactly what I was looking for, and it was. Zach's a great lecturer and everything I kinda-sorta "knew" from reading about parallelism techniques from different places is now in my bones from working on those implementations. I'm confident it will help me out a ton on my main research focus: training really competent, really small judges for post-training. Thanks Zach! I'll be back for that post-training class.

Speaking of implementations I've gotten cozy with, let's talk about the simplest and most vanilla of the parallelisms: data parallelism. Then we can talk about how to make it work if you happen to have misplaced your datacenter (DiLoCo).

# Why Scale?

We'll start with some assumptions. First, let's assume you're interested in pre-training. Lots of models on a large batch size. Second, let's assume that the model you want to train fits entirely in GPU memory, for at least one batch during training and that model is going to be trained in full precision (FP32). Let's go over what is going to need to fit into memory. Before we even start talking about activations, let's go over parameters, gradients, and optimizer states. We'll calculate all this in terms of bytes.

First, 

$$m_{params} = 4 * N$$
Each parameter is four bytes (32 bit precision). So if you're training a 7B parameter model, you've got  $4 * (7*10^9)$. There are $10^9$ bytes in a gigabyte, that's 24GB right there.

Next, you've got,

$$m_{grad} = 4 * N$$
You've got FP32 gradients for each parameter in the model during the backward pass. That's another 24GB of memory. 

Finally, you've got:
$$m_{opt}= (4 + 4) * N$$
This won't be the same for all optimizers. But let's say we're using standard [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html). Adam is going to store the momentum and variance in FP32 for each parameter. So that's an additional 48GB of memory.

So, assuming we're using FP32, we're at 96GB already, before we've even computed an activation. All that for a measly 7B parameter model. No wonder people feel complain about feeling GPU poor. 

So 7B was ambitious for fitting on a single card. I just wanted to write it out because 7B is chump change and already has you reaching for different techniques to distribute memory over multiple cards/nodes[^4]. For the purposes of this post, let's assume our model is smaller. Call it a ~1B parameter model. Those same calculations would give us 2GB for model parameters, 2GB for gradients, and 4GB for optimizers. A healthy 8GB that would fit on most consumer grade cards. It's also the size of [GPT-2 XL](https://huggingface.co/openai-community/gpt2-xl), so you're at least in the 2019 tech tree.

Now let's pick a target batch size. Our target-batch size should be at the _token_ level. OpenAI's [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165) gives us as good a place as any to start for our humble 1(.3)B parameter model.

{{< figure src="gpt-3-params.png" alt="" caption="On Teslas is crazy" >}}


A batch size of 1 million tokens. If our dataset has 1024 tokens in each sample, that means we'd want roughly:

$$\text{Number of samples} = \left\lfloor \frac{\text{Total tokens in batch}}{\text{Tokens per sequence}} \right\rfloor = \left\lfloor \frac{1 \times 10^6}{1024} \right\rfloor \approx 976$$

976 samples! Intuitively you probably understand that's not going to fit in your forward pass. But exactly how much is it _not_ going to fit in your forward pass? To really grok this we're going to need to consider activation memory, which we've been avoiding because it's slightly more complicated, and it's going to stick around through the backward pass.

The Ultrascale playbook lists it, for mixed precision with each element requiring two bytes of storage:

$$m_{act} = L\cdot seq \cdot bs \cdot h \cdot (34 + \dfrac{5 \cdot n_{heads} \cdot seq}{h})$$

$L$ is the number of layers, $seq$ is sequence length, $bs$ is batch size per sample, and $h$ is the hidden dimension of the model, $n_{heads}$ is the number of heads. 

We can simply double this in order to get to FP32. Already you can see the result is going to be quadratic with respect to sequence length, which will dominate here. Let's go ahead and fill out these values. 

$$\begin{align}
L &= 48 \text{ (n\_layer)} \\
seq &= 1024 \text{ (n\_ctx)} \\
bs &= 976 \text{ (your batch size)} \\
h &= 1600 \text{ (n\_embd)} \\
n_{heads} &= 25 \text{ (n\_head)} \\
\\
m_{act} &= L \cdot seq \cdot bs \cdot h \cdot \left(34 + \frac{5 \cdot n_{heads} \cdot seq}{h}\right) \\
\\
&= 48 \times 1024 \times 976 \times 1600 \times \left(34 + \frac{5 \times 25 \times 1024}{1600}\right) \\
\\
&= 48 \times 1024 \times 976 \times 1600 \times \left(34 + \frac{128,000}{1600}\right) \\
\\
&= 48 \times 1024 \times 976 \times 1600 \times (34 + 80) \\
\\
&= 48 \times 1024 \times 976 \times 1600 \times 114 \\
\\
&= 8,765,317,734,400 \text{ elements} \\
&\approx 8.77 \times 10^{12} \text{ elements}
\end{align}$$
Multiply by two to get into FP32, and you're looking at $17.5 \cdot 10^{12}$ bytes. That ends up being 17,500 GB of VRAM for a forward pass, or roughly 17.5 terabytes of VRAM. That's not gonna work on a single forward pass on a single card. Not on your 4090, not on an A100, not on an H100.

All that, mind you, as pre-training for a *1.5*B parameter model. They go north of a trillion in parameter count, on sequences _much_ longer than 1024 elements. So we'll need some tricks. We'll talk about two now: gradient accumulation and data parallelism. 

## Gradient Accumulation
The elites don't want you to know you don't have to called `optimizer.step()` immediately after `loss.backwards()`. You can do it whenever you feel like it!

If you've got a target batch size on a particular GPU but the activations are too large to send all of them in one go, you can break them up into _micro_-batches. Say you can only fit two samples in the forward/backward pass, but you want a batch size of eight. You can Just break up four micro-batches, successively running the forward and backward passes. Finally you can average the gradients and perform the optimizer step. 

So your _real_ batch size now looks like:

$$batch\space size = micro\space batch \space size \times gradient\space accumulation\space steps $$

So in principle, as long as you can do a forward/backward pass with at least one sample, you can increase your batch size to whatever you please while holding the memory footprint constant on our single GPU. In our example, you could run the forward/backward pass 976 times to get to the token batch size you were looking for. In principle, you could train GPT-2XL on a single consumer card!

In reality, needing to perform 976 forward/backward passes before your optimizer step is throwing some serious compute overhead down, and your wall clock time will be in terms of years. So - you could do it, but it's not what serious people do. And we're _very serious_ people. What else do we have? 

More GPUs. 
## Data Parallelism

Data parallelism is ultimately about increasing your effective batch size, similar to gradient accumulation, just with more parallel FLOPs. 

The basic idea is that we will replicate our model (which fits on a single card, remember!) onto multiple cards. Those cards could be on the same node, or cards on nodes in the same data center. If we keep the gradient accumulation steps we had before, our effective global batch size will be multiplied by the number of replicas.

$$batch\space size = num\space replicas \times micro\space batch \space size \times gradient\space accumulation\space steps $$

If you've got a target batch size, then this is a recipe for reaching it. Find out what your maximum micro batch size is, decide how many GPUs you have access to, and then fill in the gaps with gradient accumulation.

## A Brief Interlude On Distributed Torch

Before we get started, a few definitions you'll need to know as we go through code when we're talking about distributed training. You've got some arbitrary number of workers that you'd like your code to be essentially independent of. This model is called Single Program Multiple Data (SPMD). The same program is running on multiple workers with different data, and each executes independently within their own interpreters, communicating when they need to. Terms it'll be helpful to know follow:

**World Size**: This refers to the total number of processes/GPUs. So if you fired up two nodes with four GPUs a piece, the world size is 8. They are, however, zero-indexed. 

**Local Rank**: This refers to the rank within a single node. That will go from 0 to the number of gpus per node, minus one. So if you have four gpus per box, that's 0-3. 

**Rank**: This is the global rank from 0 to world size minus one. That is, 0 to 7 in this case. 

Since you're shipping the same program to multiple workers, a common pattern you'll see reading distributed torch code is a conditional to check if you're the rank 0 worker and give that one extra work. For example, if you're logging metrics in wandb, you don't want each worker in a large job doing that, you want to just have a single worker responsible for that. Regardless of what compute topology you deploy on, you'll always have a global rank 0 worker, so it's a safe grab.

If you launch through `torchrun` or `accelerate` you can get that data through environmental variables. 

```python
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
```

A full list of everything torchrun will populate in environmental variables can be found [here](https://docs.pytorch.org/docs/stable/elastic/run.html#environment-variables).

Those are your basics. Now, let's write a simple data parallelism implementation.
## Vanilla Data Parallelism

Of course, for this to work, you need to be processing your micro-batches on exact replicas of the same model. Let's write a simple wrapper that will handled vanilla distributed data parallelism for us.
```python
import torch.distributed as dist

class SimpleDistributedDataParallel:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.sync_grads = False

        for param in self.model.parameters():
            rank_0_param = param.data.clone()
            dist.broadcast(rank_0_param, src=0)
        self._sync_time = 0
        self._sync_calls = 0

```

Broadcast ensures that every local worker is going to get the same initialized parameters as our rank 0 worker. So we're off to a good start!

The next thing we need to do is make sure that each node gets different data to work with. This is trivialized thanks to `datasets.distributed`

```python
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer=get_tokenizer()
    tokenized_ds = get_tokenized_dataset(tokenizer=tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = split_dataset_by_node(
        tokenized_ds, world_size=world_size, rank=local_rank
    )
    def collate_func(batch):
        padded = tokenizer.pad(
            batch,
            padding="longest",
            max_length=None,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        padded['labels'] = padded['input_ids'].clone()
        return padded

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=collate_func,
        drop_last=True,
        shuffle=True
    )

```

Though it's a fun exercise to implement yourself. From the [documentation](https://huggingface.co/docs/datasets/en/package_reference/main_classes)

>Each node is assigned a chunk of data, e.g. rank 0 is given the first chunk of the dataset. To maximize data loading throughput, chunks are made of contiguous data on disk if possible.

So from my entire dataset, each node is going to be assigned a certain number of samples from that dataset, and this will be invisible to me when I'm iterating through my dataloader.

Now we've ensured that our replicas _start_ in the same place and that when they process data it will be different data that gives us unique gradient information. Now we've got to be able to sync our gradients between workers before the optimizer step. In addition, we want to ensure that `backwards()` does not _always_ sync gradients, because gradient accumulation means we may be calling `backwards()` several times before we're actually ready to run the optimization step. 

Also, I want to make sure we can measure the communication time for syncing the gradients. But that'll be important later. 

```python
class SimpleDistributedDataParallel:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.sync_grads = False

        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        self._sync_time = 0
        self._sync_calls = 0

    def sync_gradients(self):
        """
        Call before optimizer step
        """
        if not self.sync_grads:
            return
        t0 = time.perf_counter()
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self._sync_time += t1 - t0
        self._sync_calls += 1
        
    @property
    def avg_sync_time(self):
        return self._sync_time / self._sync_calls if self._sync_calls > 0 else 0
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def disable_grad_sync(self):
        self.sync_grads = False

    def enable_grad_sync(self):
        self.sync_grads = True

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
```

Mostly this is a wrapper around our model. Most of the api, like `__call__`, `train` and `eval` we want to keep the same. 

The big thing here is `sync_gradients`. Once we've reached our desired number of gradient accumulation steps, we want to make sure the replicas have a shared understanding of the gradients before the optimizer step runs. To do that, we want to do an all-reduce, where the data is distributed between workers with some function applied to it. In our case, that'll be averaging. At the end of the operation each replica will have the same understanding of the gradients.

```python
model.train()
num_batches = 0
for (i, batch) in enumerate(train_dataloader):
	batch = {k: v.to(device) for k, v in batch.items()}
	if i > 2048:
		break
	if (i + 1) % gradient_accumulation_steps == 0:
		dp_model.enable_grad_sync()
	else:
		dp_model.disable_grad_sync()
	output = dp_model(**batch)
	loss = output.loss / gradient_accumulation_steps
	output.loss.backward()
	if dp_model.sync_grads:
		dp_model.sync_gradients()
		optimizer.step()
		optimizer.zero_grad()
		if global_rank == 0:
			wandb.log({"loss": loss.item() * gradient_accumulation_steps, "step": i, "avg_sync_time_seconds": dp_model.avg_sync_time, "perplexity": torch.exp(loss).item()})
			num_batches += 1

```

With that written up and some standard dataloader code written around it (which you can look at [here](https://github.com/SJCaldwell/naive-data-parallel/blob/main/sillydp/main.py) if you're interested) we've implemented a very basic data parallelism.

Some profiling here:

{{< figure src="vanilla_avg_sync_in_seconds.png" alt="" caption="Shocking: distributed nodes take longer." >}}

We on average do the all-reduce comms in about 200ms. This is quite high for GPUs on the same node! But that mostly has to do with using our hand-rolled algorithm that doesn't support bucketing. 

The devil is here:

```python
for param in self.model.parameters():
	if param.grad is not None:
		dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
```
We're generating a lot of overhead. This is because for _every_ single parameter, we're calling an all-reduce. Each of these is separate, so there's some overhead in setting up communications being done each and every time. Even if all the data is _small_ it's a lot to do. Instead, we could use a bucketing strategy. This breaks the data up into larger chunks, combining data of up to about ~25MB together. This reduces the total amount of communications that need to get done. You can see a good implementation of this over [in picotron](https://github.com/huggingface/picotron/blob/main/picotron/data_parallel/data_parallel.py).

We won't implement it here, because we're interested in a different question. Let's hold that operation constant, and instead play with how long it takes to perform as we pull these two workers further from each other. 

If I run the exact same code but on two different nodes without Remote Direct Memory Access (RDMA) it runs in about ~500ms. Worse still, but tolerable.

We can keep extending that distance, just based on what we know about the internet. We're transferring on the order of 18MB with each all-reduce here. On the same node, with PCIe we've got a bandwidth of around ~10-25 Gbps. Latency will be short. On different nodes we've got to kick on the network stack which increases our overhead (thus the 500ms). Not so bad.

But what if we don't _have_ nodes on the same rack? What if they're not even in the same data center? What if we don't _have_ a data center, and are instead sourcing compute from wherever we can get it?

{{< figure src="smoke_em_if_you_got_em.png" alt="" caption="Pic related: the wherever we can get it" >}}

In this world, we may be pushing those 18MB over regular old internet bandwidth. That might take the all-reduce to ~20 seconds. This is all with a relatively small model, and that parameter count and the gradients that have to be moved can get quite a bit larger as you scale the size of your model and the number of machines that have to communicate. 

You want to train a big model. You might even have the dollars to spend on spot-compute. But you're data center poor and you want to do research with the big boys. What do you do?
## DiLoCo - Take What You Can Get

So, our compute isn't shared in a single data center, but rather plucked from discrete nodes and clusters located all over the continent - or the world. We want to do data parallelism to increase our effective batch size, but it seems very likely if we use our current approach GPUs will spend most of their time idling due to expensive and slow network operations.

Our ideal technique would be one that's stackable (in that it uses data parallelism but does not prevent using other parallelisms), comfortable with heterogenous compute (different nodes/clusters with different GPUs), capable of communicating infrequently across a large geographic distance. Since we're GPU-poor and use spot instances, it would also save us a lot of gray hair if it was tolerant of nodes dropping out or joining partway through training. 

As it turns out, that exists. It's called [DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105).
## The DiLoCo paper
If you've done a brief read of HuggingFace's [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)[^3], the DiLoCo paper is actually quite readable.

The basic setup mirrors data parallelism exactly. You've got replicas of your model, deployed to potentially heterogenous compute. Just like standard data parallelism, each replica also has its own discrete part of the dataset available for training. 

What's different is that each replica also saves the initial state of the model before training begins, and each replica has _two_. That initial state of the model copy is offloaded onto CPU, since it won't be used frequently. Onto the optimizers: the first is called the "inner optimizer". It's a very standard AdamW optimizer in the paper, but it's whatever you would use for standard training. The inner optimizer loop is entirely normal, and does no communication between workers, and so does not incur any communication cost. You can add gradient accumulation as you like, whatever you need to get to an effective batch size you want for training. 

In addition, training proceeds completely normally for a set amount of _inner steps_ (let's call it $H$). Training proceeds independently for all nodes. $H$ is a hyperparameter, but to be useful it's set at something on the order of 500. That is, you're calling `optimizer.step()` on the inner optimizer 500 times before any communication happens between these disparate hosts. 

So essentially you're training $n$ replicas of the model, one for each worker, starting from the same place and diverging as they update. How does this bubble up to a single trained model at the end? 

The outer optimizer is responsible for that. Every $H$ steps, the outer optimizer loop happens. This is the tricky bit.

The outer optimization step collects *psuedo-gradients* by looking at the difference between the original weights it had the last time the outer optimizer was called. At the first step, this was be the pre-trained weights or the initial values of the weights. The psuedo gradients are `initial_parameter - replica_parameter` for each parameter in the neural network. These psuedo gradients are different for each worker, since they've all been trained on different data and have been trained independently for these 500-odd steps.

An all-reduce is called on this step, so each worker averages these psuedo-gradients before calling the outer optimization step. 

This outer optimizer is attached to the same weights as the inner optimizer, so when `outer_optimizer.step()` is called, each replica of the weights will be updated from the initial values with the same psuedo-gradients. So the replicas have once again been synced. A new copy of these weights is now stored in CPU for the next outer optimizer step, and training continues. 

The inner-optimizer is not reset, so while each replica has the same weights, their AdamW keeps its first and second moment estimates. This results in transient training spikes, but doesn't cause a problem otherwise. Training continues until the desired amount of outer steps have been reached.

If this sounds miraculous and unlikely, it's probably because of your intuition about AdamW. AdamW is not what the outer optimizer is using. The paper very specifically uses [Nesterov](https://machinelearningmastery.com/gradient-descent-with-nesterov-momentum-from-scratch/).

Specifically, the paper says:

>We hypothesize that the Nesterov’s gradient correction is particularly helpful with the outer gradient that span hundred of training steps.

An intuition about this is [intuition].

The paper ends with a series of ablations. What if compute joins or leaves during training? They find models end up generalizing well given a fixed compute budget, regardless of how that compute is made available over time. What if we do all of this on a single worker? Convergence speeds up. What if the communication is asynchronous and spotty, and outer gradient communications don't always reach a given worker? No problem, let the worker continue training the model for another round of $H$ inner-states and try again, it only slightly effects the final perplexity of the model. In general, the paper concludes that DiLoCo is just a vary robust algorithm for data parallelism.

This isn't pure research, either. Prime Intellect took it out on the road with [Intellect-1](https://www.primeintellect.ai/blog/intellect-1-release). Prime Intellect's training used their own DiLoCo implementation that supports [FSDP2](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp1_vs_fsdp2). DiLoCo is used across nodes and FSDP within nodes. The resulting 10B parameter model converged, training on 14 concurrent nodes on three different continents, across 30 different compute providers.

A cool detail in the paper is that the all-reduce operation during the outer optimizer step took between one and seven minutes. This occurred after the inner optimization step roughly every 38 minutes. They chose $H$ to be a somewhat conservative 100 steps. This means that without DiLoCo, the all-reduce would've needed to be incurred for every one of those 100 steps. That would mean roughly every 23 seconds a lag of 1-7 minutes would've been introduced! Training would've been totally infeasible.

Now that we're sufficiently motivated to understand _how cool_ it is, let's implement a vanilla DiLoCo and see how it works. 
## Implementation

We're going to create a wrapper the same way we did it for vanilla data parallelism, with some tweaks. 

```python
class Diloco:
    def __init__(self, 
        model, 
        inner_optimizer, 
        outer_optimizer, 
        warmup_steps, 
        total_steps,
        inner_steps: int = 100, 
        outer_steps: int = 10
    ):
        self.model = model
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer
        self.scheduler = get_cosine_schedule_with_warmup(self.inner_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        self.offloaded_last_sync_parameters = self._get_offloaded_parameters()
    
```

We'll distribute our initial weights the same way, again. We'll now need an `inner_optimizer` and an `outer_optimizer`, so we'll grab both of those.

After we've synced, we want to offload our starting state into `self.offloaded_last_sync_parameters`. This will always be the current state of the model. Our inner optimizers run on and modify our specific replica trained on their own data. When we're ready for the outer step, we'll need the most recent synced copy of the parameters. We offloaded these to CPU to avoid keeping another copy in GPU vram.

```python
def _get_offloaded_parameters(self):
	return [
		param.data.detach().clone().to("cpu")
		for group in self.outer_optimizer.param_groups
		for param in group["params"]
	]
```

What used to be just `step` on the replicas in data parallelism is now our `inner_step`. Our inner step doesn't change very much. Though, this time I applied gradient clipping for smoother training, as well as a learning rate schedule since these are included in the paper. 

```python
def inner_step(self):
	torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
	self.inner_optimizer.step()
	self.scheduler.step()
	self.inner_optimizer.zero_grad()
```

The outer step is where things get properly interesting. Let's look at the conditions that cause it to fire, and then look at the implementation itself.

```python
# ... normal train_dataloader setup
for (i, batch) in enumerate(train_dataloader):
	real_step = (i + 1) // gradient_accumulation_steps
	batch = {k: v.to(device) for k, v in batch.items()}
	output = diloco_model(**batch)
	loss = output.loss / gradient_accumulation_steps
	output.loss.backward()
	if (i + 1) % gradient_accumulation_steps == 0:
		diloco_model.inner_step()

		if real_step % inner_steps == 0:
			diloco_model.outer_step()
```
So we still have our gradient accumulation steps for the inner optimizer, and only called `inner_step` when we've accumulated enough gradients to hit the batch size we're interested in. 

After we've called our inner step, we check to see whether we've hit the proper number of `inner_steps`. This is the $H$ we discussed above. If we have, it's time to call the outer step. 

```python
def outer_step(self) -> None:
	"""
	Outer step for Diloco.
	Loads last sync parameters from CPU to GPU and 
	  computes the psuedo-gradient for outer optimizer.
	Updates the offloaded parameters to CPU.
	"""
	replica_params = [
		param
		for group in self.inner_optimizer.param_groups
		for param in group["params"]
	]

	for replica_param, last_sync_param in zip(replica_params, self.offloaded_last_sync_parameters):
		last_sync_param_on_device = last_sync_param.to(replica_param.device)
		replica_param.grad = last_sync_param_on_device - replica_param.data
		dist.all_reduce(tensor=replica_param.grad, op=dist.ReduceOp.AVG)
		replica_param.data = last_sync_param_on_device
	
	self.outer_optimizer.step()
	self.outer_optimizer.zero_grad()
	self.offloaded_last_sync_parameters = self._get_offloaded_parameters()
```

First we get our current replica parameters so they can be zipped against our last synced ones. `replica_param` is the current state of the model that's already loaded into GPU memory we've been optimizing.

First, briefly, we read the offloaded parameters into GPU memory. At this point, there aren't any activations being computed, so the storing the additional model on device briefly isn't super painful. Then, you set the recently zeroed gradient of the on device model to the difference between the last sync, and the replicas current understanding of the world. This distance becomes the psuedo gradient. Then, you perform an all-reduce, so the gradients now represent the average distance between the last synced model and its replicas. Finally, you place the last synced model parameter data over the replicas weights.

So very briefly, you overwrite all of your progress. The models are right back to their starting state! But crucially, they're back to their starting state with our psuedo-gradient information. All that's left to do is take your step. As soon as that step occurs on each replica, the models have made a large update with information from the training they each did independently. Finally, you overwrite the last sync parameters by offloading the new weights you've computed into CPU for the next step, and continue training as normal.

## Results

{{< figure src="diloco_train.png" alt="" caption="It runs!" >}}


If you want to try this for yourself, you can check out my repo here for [NanoDiloco](https://github.com/SJCaldwell/NanoDiloco). The wandb logs are [here](https://wandb.ai/sjcaldwell/nano-dilco/workspace?nw=nwusersjcaldwell). If you want to see what production grade DiLoCo looks like, Prime Intellect has a beautiful repo for it [here](https://github.com/PrimeIntellect-ai/prime/blob/de5b9317da5f57247f11abed2fca259076795460/src/zeroband/diloco.py#L24).

And to Zach, who I hope reads this: great class! I've found in general it's easy to find teachers if the thing you want to learn is ~5 years out of date, but the closer you get to the cutting edge the rarer it is to find someone who is both gifted at doing the work itself as well as concisely communicating that understanding to other people, pulling all the pedagogical knobs and levers required. That's why Karpathy gets so much love. I think you've got the same gift[^6]. 


[^1]: "Look at the crummy latency on this model, look at the dumb mistakes it makes, and how limited its labels are! We could never go to prod with this. Now, let's talk about our labeling budget..."

[^2]: That age-old ill. Eventually I will be old enough where I stop doing it.  

[^3]: And if you haven't, you should! It's an excellent introduction to different parallelism strategies necessary for training modern models.

[^4]: Or just placing your faith in [Unsloth](https://github.com/unslothai/unsloth) and riding that single GPU as far as it'll go

[^5]: OpenAI did this on Tesla V100s, which had 32GB of memory. Some versions had 16GB. 
