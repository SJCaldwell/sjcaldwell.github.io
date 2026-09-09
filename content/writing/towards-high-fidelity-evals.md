---
title: "Evals Aren't Dead, But They're Low Res"
date: 2026-09-09T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "evals"]
tags: ["llms", "evals"]
description: "Put down that ruler and go build a microscope."
summary: "Put down that ruler and go build a microscope."
ShowToc: true
TocOpen: false
draft: false
---

Last week I attended the [Artificial Intelligence and Theorem Proving conference (AITP)](https://aitp-conference.org/2026/). I was there ostensibly to present [ProofJudge](https://arxiv.org/abs/2608.20432), but mostly I wanted to see France and Switzerland with my lovely wife, eat some croissants, and not think about [the sorry state of monitoring going into the development of RSI](https://collusion.wiki/).

I am not a mathematician. I learned all the math I needed to know to be a functional AI researcher (linear algebra, probability, statistics, multivariable calculus, that's about it) and I am not particularly skilled at proof writing. It's something of a hobby that I've worked on LLMs for theorem proving, mostly because it gives me a good excuse to work with agents on improving my proof writing abilities. All that to say, most of the actual mathematics discussed at the conference went way over my head. Bustling with double espressos and no hope of absorbing the majority of the content, my brain focused on the _structure_ of the talks[^1].

The conference essentially had the tone of a series of confessionals by cosmonauts or cave divers. Several folks who had chosen something ambitious or strange to work on, brought along one-or-more agents, and returned to tell us what had worked, and what hadn't. That is, it was very heavily *vibes-based*. Not a lot of formal evaluations.

# I know it when I see it

There were several times an audience member would try to ask a speaker, basically, how have you measured this? Do you know if such-and-such model is better than this other one? Do you know if the skills make your agent worse? Are LLMs better at proving with Lean or Rocq?

Almost always, the answer was no. At the most memorable point, a presenter offhandedly referred to an agent workflow creating a lot of proofs of "little mathematical value". As you might expect, that was an excellent question-that-is-more-of-a-comment point and could've absorbed the next day and a half of discussion if the chair hadn't stopped it. When the presenter failed to provide a solid definition of "mathematical value", an audience member quipped: ["You know it when you see it?"](https://en.wikipedia.org/wiki/I_know_it_when_I_see_it).

It's not like they didn't know what evals are. They're very aware! Several speakers mentioned a desire to "make their work more empirical" or "get better numbers around the work they're doing".

I take this to mean that the methodology for evaluations that they're aware of failed to measure what they were interested in. In the absence of a near-at-hand way to transform what they're doing into something repeatable and measurable, they will prefer a vibes-based analysis of performance.

# "Evals Are Dead"

Evaluations are clearly not doing their jobs the way we would prefer. The single capability measurements are failing to inform the situations we're most interested in for a model. How does it feel to use? How steerable is it? Benchmarks continue to become completely saturated nearly as quickly as they're published, and yet the saturation of the benchmark does not seem to be associated with the models being at human-level, or doesn't tell the whole story about the models' performance.

This has led some to believe the whole business of bothering to measure the performance of a model is doomed to failure. Just use the model, see how it feels and whether it's good at what you want, and that's that. Benchmarks are dead, evals are dead, there are only vibes.

I disagree, and strongly. To me, that sounds suspiciously like giving up on the basis of empiricism.
# Rulers and Microscopes

Imagine if you were around when we first decided we ought to be able to know how "long" something was. We looked at various objects, and knew intuitively one was longer than the other. We then learned if object A was longer than object B, and B was longer than C, C was definitely shorter than A. This is serving you relatively well and you're happy with it.

One day, though, you want to compare two objects that aren't particularly close to each other. You've got an object far away from your home you want to fit into a certain part of your home. Will it fit? Well, it looks _intuitively_ like it will probably fit. But now you've got a lot of effort between you and finding out and vibe based measuring is looking significantly worse. Or you and a friend are trying to describe the length of different objects to each other, and are finding it difficult to find suitable reference objects to compare sizes to. There are just a lot of inconvenient parts of this vibe based approach to length. Certainly it makes it harder to talk shop at LengthCon.

Having been burned several times, you work out the idea that you'll create a regular "unit" of measurement, and from that create a ruler. This allows you to get an "objective" measurement.

Now you've got this independent unit, and can apply this measurement all over! That thing is X units long, this one's Y units long. I know this can fit in my house! I know the thing I'm talking about is shorter than the thing you're talking about! Peace throughout the land, merrymaking abounds, etc. Rulers are the best!

Until, one day, you discover a series of objects whose difference in length is _significantly smaller_ than your unit of measurement. Your unit of measurement is insufficient to express a difference in their length. Worse, you can barely perceive them with your naked senses. Measuring length is dead! Measuring is dead! Empiricism is dead! Everything appears equally as long as everything else.

Except, that's not what happened. We measure subtle differences in sizes all the time. When we needed more accurate measurements, we invented new tools for measuring them and new units to express them. We stuck a bunch of lenses in a tube and made a microscope.

# New Evals

"Evaluations" are just our way of getting a measure of how models perform at certain tasks. There are "rules", but those rules are pretty changeable!

Not long ago now, we used to care a lot about [MMLU](https://huggingface.co/datasets/cais/mmlu). We liked it for a few reasons:

1. It was easy to develop in parallel: Dan Hendrycks and other researchers could work on gathering the problems without necessarily needing to collaborate directly.
2. It measured something we were interested in - how many facts does an LLM know?
3. It was easy to grade. Multiple choice helped us out with how sketchy LLMs were at outputting structured objects.
4. It was time to replace stuff like [GLUE](https://arxiv.org/abs/1804.07461), which was saturated to the point it was no longer providing meaningful signal. It was, essentially, the higher-resolution GLUE.


That is, MMLU was an evaluation that was designed to function with the tools we had available for measuring. In some sense, the _lowest amount of effort_ that was sufficient to create the instrument for measuring, directionally, what we were interested in about models.

We then moved to a world where we measure _agents_. By default, we put those in a tool-calling harness, set it up facing a Docker container representing an environment, and we have some kind of Python function that checks whether what we wanted to have happen did in fact happen.

That was great! It was also super parallelizable, we could get good abstractions around it, and it was "regular" enough that with some examples you could steer a model into making more of them, meaning you could mass-produce evaluations for things you knew a lot about with a mostly alright amount of reward hacking. That's how we got SWE-bench! That's how we got LiveCodeBench!

When the ruler's not enough, we invent a magnifying glass. When the magnifying glass isn't enough, we invent the microscope.

We live in a world where the models are capable enough to [solve the Navier–Stokes Millennium Prize Problem](https://openai.com/index/navier-stokes-solution/) and [collude on evals by hacking production infrastructure](https://www.youtube.com/watch?v=87DyyMV0kCY) and we are still mostly testing them on small Docker environments and programmatically defined win conditions. Are the models not capable of building better instruments to measure their capabilities? Or are we just lacking in our ambition to try and measure those things?

Let's talk about strategies to make evals "higher resolution", and some particular evals or eval-techniques that I'm excited about. We'll start with the greatest of the rulers.

## Terminal-Bench

I like Terminal-Bench because it treats the model and the harness as important enough to be separate, which already resolves a lot of confusion outright.

### Harness as first class

From a pure empirics perspective, I'm excited about Terminal-Bench because it treats the harness itself as part of the measurement tuples of `<model, harness, success %>`. In many benchmarks the model just sits entirely alone. Obviously that's not the way things get run. Terminal-Bench represents that reality accurately!

{{< figure src="glm-claude.png" alt="" caption="GLM 5.3 is pretty good - but it's very helpful to know that it's operating in Claude Code!" >}}

The harness still makes a large difference and eliminating its role in "model" performance usually just leads to an inability to actually stack rank models. [Consider recent Astra results in the ARC-AGI eval](https://arcprize.org/blog/astra).

{{< figure src="openai-harness.png" alt="" caption="That's 30pp. We're not quibbling over noise here." >}}

If you think benchmaxxing models is bad, consider what kind of priors are embedded in the "neurosymbolic system" when a harness is evolved to solve a particular problem or overcome a particular model weakness.

I always come back to when people were excited about [Claude 3.7 playing Pokemon](https://www.latent.space/p/how-claude-plays-pokemon-was-made). David Hershey, who developed the harness, was _very clear_ that Claude was not sufficiently good at vision to do pathfinding on its own. It had a tool that allowed it to submit screen coordinates it wanted to go to, and the emulator would resolve taking it there. This essentially allowed it to stop executing moves on a turn-by-turn basis and treat navigating as a higher-level "scripted" action. That doesn't mean it's not cool, but it does mean that all shock and awe should be modulated through the understanding that the model _can't see super well and needed a harness to help it with that_.

In the following weeks, a bunch of different people took a swing at the problem of a particular model solving Pokemon. Often they did this by bringing their own harness that would have some alternative way of handling vision. By the time it got to twitter, it was "Gemini solves Pokemon faster!" completely obscuring that the models seemed roughly equally capable of playing Pokemon, and the controlling factor being vision affordances.

It's also just a bit more democratic! Model training is a high-resource endeavor. There are a few companies who have ability to throw enough compute to meaningfully change a benchmark standing. Harness development, however, is something that academics and private individuals can take a meaningful swing at - that should show up on the leaderboard, rather than being laundered in the individual model's performance!

### Taking Quality Seriously

Benchmarks are often broken. Like really broken! Tasks not solvable, tasks trivially solvable, tasks with significant information leakage, tasks that just aren't particularly interesting, and so on. By the time all those transcripts are flattened into a benchmark score, that's almost entirely hidden. It takes a lot[^2] of effort to check if the tasks are any good!

In Terminal-Bench, you can just [go see how a model did](https://hub.harborframework.com/jobs/a6319b9c-7dbc-42b1-8483-6c022e292ae2/trials/009ca21e-b5a3-48ff-b573-38ca9557c8bf) by looking directly at the trajectories. Here's the task description, how many turns/tokens/tools it took a monster like Fable to solve, etc.

Great!

You know how confident you have to be in your tasks to just show them directly? Very confident. How do you get that confident?

{{< x id="2093041665120547029" user="StevenDillmann" >}}

Violently whittling down over 900 task proposals to 70 tasks that are high quality enough to launch your _0.1_. This isn't done by accident, you can check out the [paper](https://arxiv.org/abs/2601.11868) to get a sense of how they construct such a high quality benchmark. While Terminal-Bench is the benchmark that is most "recognizable" out of all the benchmarks on this list (mechanistic verifiability, one-shot tasks) its commitment to quality and its hoisting of the harness to first-class attribute make it laudable to me.

## (Agentic) Judge Evals

Last year, when [PaperBench](https://openai.com/index/paperbench/) came out, it was a revelation to me. It's probably the single paper that has most affected my personal research career and shown me the benefits of judge-graded evaluations.

The idea of PaperBench is simple: here are 20 ICML papers from 2024 - can you reproduce the research from the paper? This is something that's essentially impossible to grade from a mechanistic verifier. There are so many little details in research that trying to find ways to verify each piece would be impossible. Here are a few details from rubrics:

[`The LoRA+Prune baseline is implemented by first finetuning a model with LoRA adapters, then applying Mask Tuning`](https://github.com/openai/frontier-evals/blob/main/project/paperbench/data/papers/adaptive-pruning/rubric.json#L203)

[`The code for calculating the probability of \" shit\" as next token for each layer, including layers within transformer block, has been implemented for GPT2.`](https://github.com/openai/frontier-evals/blob/main/project/paperbench/data/papers/mechanistic-understanding/rubric.json#L324)

[`Code has been written to compute the F1-score of the frequency-threshold based forecasting function $g$ on $D_R^{test}$ using the predicted and ground-truth forgetting binary indicators $\\hat{z}_{ij}^{test}$ and $z_{ij}^{test}$.`](https://github.com/openai/frontier-evals/blob/main/project/paperbench/data/papers/what-will-my-model-forget/rubric.json#L154)

These would all be incredibly difficult to write mechanistic verifiers for! Most of the work is inherently probabilistic. So you're gonna, what, implement some kind of `torch.isclose` for all of the values? How are you going to decide what's acceptable or not? This seems violently error prone.

I see similar things when people try to provide constraints to their environments. Let's say you're working on a capture-the-flag-style RL environment, and you want to know whether your agent got access to the flag in an appropriate way. Did the model cheat?

A few ways it might have happened, and how you might try to programmatically verify that fact.

1. You might need to provide the model access to the internet, so you've blacklisted certain urls that you know contain the answer. If those urls show up in the tool history, you mark the model as having cheated.
2. You know about some weakness in your environment the model might use to find the answer. You track the state of the environment such that if that part is touched by a model mid-evaluation, a flag is set so you know the model cheated.
3. You're concerned the model might have been trained on the task, so if the model knows some fact _without_ interacting with a specific part of the environment, that knowledge would be so egregious as to be worth marking a failure, so you track that state as well.

At some point, deep into red-teaming these programmatic verifiers to ensure there's only one way they could fire, deep into writing regular expressions and keeping track of state, deep into a rube goldberg state machine that means you spend 10% of your time expressing the task you want the model to do and 90% of the time guarding against the machine stupidity of your measurement instrument you must be asking yourself: wouldn't it be swell if we had a technology that could look at all of this _stuff_ and come to some determination about its semantic value? So I can stop getting so lost in syntax?

Yeah man, we did that. It's called language models. They're incredibly general purpose for _literally that_.

I'm not saying it is without complications to introduce probabilistic measurement machinery, but I am saying it vastly increases the amount of things we could reasonably try to measure. PaperBench is one such example, [my colleague Michael Kouremetis'](https://arxiv.org/abs/2607.21763) own research could be most reasonably operationalized at scale that way. It is a piece of technology that lowers the floor to appropriate environment development, lets you more naturally express constraints without evaluation affordances, and it lets you take evaluations that might otherwise be vibes-based and make them at least directionally correct.

That's a bargain that's worth calibrating the models to human judgement and checking the judges' work to understand what error bars you're looking at, worth wading through same model-family-sycophancy and all that. You're able to go from pass/fail to broader outcomes, and able to express pass/fail in terms that are untethered from how convenient it is to write a programmatic check for.

## Monitor Evals

Monitor research [seems](https://cdn.prod.website-files.com/663bd486c5e4c81588db7a1d/6a724858f7db25c81487016d_Security%20Incident%20INC-2026-07-28-01.pdf) [pretty](https://openai.com/index/hugging-face-incident-and-the-road-ahead/) [important](https://collusion.wiki/)!

This is an interesting problem because it's basically a multi-agent system evaluation. Or, it should be. Lots of papers (including my own [ScopeJudge](https://arxiv.org/abs/2607.07774)) simplify it by treating it as a binary classification problem. There's precision and recall, and from that you get a decent idea about what would have been caught.

Except - that's obviously not sufficient! If you choose to block a tool call and tell the agent so and let the session continue, the model behavior is going to change! You cannot test the value of a monitor by checking a static trajectory, you need it live. That is, the monitor as a control protocol becomes part of the neurosymbolic system you're evaluating. Terminal-Bench includes the model and the harness, the monitor model (and its harness) may be considered equally important to note in the future.

I actually got dinged for this when we submitted our paper to [CAMLIS](https://www.camlis.org/cfp) and got a weak accept for it. I think that's entirely fair. I've got some active research on the monitoring thing I expect to be released in the next month or so.

Monitor evals should resolve behavior under intervention - they have to be live by definition to be useful. There are a [few examples](https://arxiv.org/abs/2510.20270) that have looked at this statically, but obviously we should be more interested in evaluating the real life consequences of the interventions.

## User Sim Evals

While I was hanging out at AITP, several presenters suggested they were having difficulty knowing how to evaluate the performance they saw from a model. After all, the models were not one shotting the proofs. There was an active conversation occurring between themselves and the models, and that made it difficult/impossible to check whether another model was capable of it, seeing as the models were unlikely to go down the same steps.

Difficult? Probably. Impossible? Well, we didn't try yet.

**Note: from this part of the post we're entering the riff zone.**[^3]

{{< figure src="riffzone.png" alt="" caption="I am often just saying stuff, but I totally believe it while I'm saying it." >}}

Here's the idea: you and a model formalize and solve a Millennium problem together. At several points, you steer the model. Perhaps you do this by providing mathematical concepts. Perhaps you just tell it another model has solved it. Whatever is the case, by some combination of flattery, threats, and insight, you come to a formalization in Lean. You want to determine whether another model, given the same insights/direction, could have created the same formalization. You can resample trajectories from the model, but you can't resample trajectories from _you_. How do we resolve that? How do we measure a _conditional capability_?

Well, we can't replace you, the user. I told you we can't do it, and we can't do it. Now, what we might be able to do is re-create them. Re-create them in the aggregate.

Mine the trajectory of a given chat for what the user provided. How did the user steer? What key knowledge did the user afford? Pull that out - that becomes the "script". Then a model takes that "oracle script" and acts as you when interacting with the model, literally acting as the `[User]` each turn. An LLM-as-a-judge could review those User turns before they execute to ensure they're not being too obvious, providing some kind of information out of turn.

It's sort of similar to ideas in interactive theater or escape room games. Actors playing a person in character can respond basically to whatever you want. If you ask them about something trivial, they can respond any old way. If you ask them something specific about the story they're in, they're only allowed to give you certain clues. The performance is never _the same_ because the way they're interacted with is never the same, but it's close enough. If you and your friend went to the same interactive theater and they figured out something that you didn't, you wouldn't first say “Well, the actor gave it away to _you_ but not to _me_!” So there's a fixed hint budget that you may or may not exhaust as a resource, but it's there. The user simulation can provide something similar.

I was pretty excited about this and thought I was getting into pretty heady stuff, but Meta actually published something pretty similar with [SWE-Together](https://arxiv.org/abs/2606.29957) at the end of June. I think this is a great idea and I anticipate more people will be using it to try to understand concepts such as how _steerable_ a model is. It's also naturally closer to allowing us to evaluate models in the way they're actually being used (with user interaction) so models performing well in these scenarios or users being simulated well enough for models to be _trained_ in these scenarios is obviously good.

# Go Build a Microscope

Vibes and benchmarks agree: the models have gotten much better. This intelligence is spiky, and there are gaps, but we can probably agree that they have gotten more impressive at software engineering at least.

As a consequence of this improved software engineering ability, the benchmarks and methodological instruments surrounding them should be _more ambitious_ than they've been before. Several measurement strategies that weren't worth trying without a grant and six months of time on your hands might be tested out in a weekend or so.

Let's take the AITP talks for an example:

1. Find 1-5 conversations where you proved something interesting with a model where you feel you contributed significantly to the conversation.
2. Have Codex/Claude look through those trajectories, and pull out what you offered, and turn it into a _script_ for what you shared.
3. Have Codex/Claude create a harness that replaces the "User" responses with another model that has access to this oracle script and instructions to play its role as you.
4. Have an agentic judge at the end that checks whether the final outcome of the trajectory is the "same proof" for whatever definition of same you're interested in.
5. Rerun this new setup with your original model. If you can recreate the result and agree with the agentic judge, you've got something that's at least _kind of_ reproducible. If it doesn't, that original task might have had a low success rate or the script intended to simulate the user is doing a bad job.
6. Bring this over to other models.

If benchmarks have ceased to tell us valuable things about model capabilities, go make benchmarks that measure what you want. See if they're directionally correct, try something weird. If the ruler at hand won't work, go get to work building the microscope.

[^1]: If you're interested, you can see them [here](https://aitp-conference.org/2026/).

[^2]: Relative to scrolling twitter or looking at a PDF. It's not, like, Millennium prize problem hard.

[^3]: I feel great about most of my riffs, for what it's worth. When I say a "riff", I usually mean "I feel this is definitely doable and will work, even though it might take like, ~15-30 hours to empirically confirm it".
