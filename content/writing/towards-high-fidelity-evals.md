---
title: "Evals Aren't Dead: They Need to Grow Up"
date: 2025-08-17T00:00:00Z
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

I am not a mathematician. I learned all the math I needed to know to be a functional AI researcher (linear algebra, probability, statistics, multivariable calculus, that's about it) and I am not particularly skilled at proofwriting. It's something of a hobby that I've worked on LLMs for theorem proving, mostly because it gives me a good excuse to work with agents on improving my proof writing abilities. All that to say, most of the actual mathematics discussed at the conference went way over my head. Bustling with double espressos and no hope of absorbing the majority of the content, my brain focused on the _structure_ of the talks[^1].

The conference essentially had the tone of a series of confessionals by cosmonauts or cave divers. Several folks who had chosen something ambitious or strange to work on, brought along one-or-more agents, and returned to tell us what had worked, and what hadn't. That is, it was very heavily *vibes-based*. Not a lot of formal evaluations.

# I know it when I see it

There were several times an audience member would try to ask a speaker, basically, how have you measured this? Do you know if such-and-such model is better than this other one? Do you know if the skills make your agent worse? Are LLMs better at proving with Lean or Roqc? 

Almost always, the answer was no. At the most memorable point, a presenter offhandledy referred to an agent workflow creating a lot of proofs of "little mathematical value". As you might expect, that was an excellent question-that-is-more-of-a-comment point and could've absorbed the next day and a half of discussion if the chair hadn't stopped it. When the presenter failed to provide a solid definition of "mathematical value", an audience member quipped: ["You know it when you see it?"](https://en.wikipedia.org/wiki/I_know_it_when_I_see_it).

It's not like they didn't know what evals are. They're very aware! Several speakers mentioned a desire to "make their work more empirical" or "get better numbers around the work they're doing".

I take this to mean that the methodology for evaluations that they're aware of failed to measure what they were interested in. In the absence of a near-at-hand way to transform what they're doing into something repeatable and measurable, they will prefer a vibes based analysis of performance.

# "Evals Are Dead"

Evaluations are clearly not doing their jobs the way we would prefer. The single capability measurements are failing to inform the situations we're most interested in for a model. How does it feel to use? How steerable is it? Benchmarks continue to become completely saturated nearly as quickly as they're published, and yet the saturation of the benchmark does not seem to be associated with the models being at human-level, or doesn't tell the whole story about the models performance.

This has led some to believe the whole business of bothering to measure the performance of a model is doomed to failure. Just use the model, see how it feels and whether it's good at what you want, and that's that. Benchmarks are dead, evals are dead, there are only vibes. 

I disagree, and strongly. To me, that sounds suspiciously like giving up on the basis of empiricism.

# Rulers and Microscopes

Imagine if you were around when we first decided we ought to be able to know how "long" something was. We looked at various objects, and knew intuitively one was longer than the other. We then learned if object A was longer than object B, and B was longer than C, C was definitely longer than A. This is serving you relatively well and you're happy with it.

One day, though, you want to compare two objects that aren't particularly close to each other. You've got an object far away from your home you want to fit into a certain part of your home. Will it fit? Well, it looks _intuitively_ like it will probably fit. But now you've got a lot of effort between you and finding out and vibe based measuring is looking significantly worse. Or you and a friend are trying to describe the length of different objects to each other, and are finding it difficult to find suitable reference objects to compare sizes to. There are just a lot of inconvienent parts of this vibe based approach to length. 

Having been burned several times, you work out the idea that you'll create a regular "unit" of measurement, and from that create a ruler. This allows you to get an "objective" measurement.

Now you've got this independent unit, and can apply this measurement all over! That thing is X units long, this ones Y units long. I know this can fit in my house! I know the thing I'm talking about is shorter than the thing you're talking about! Peace throughout the land, merrymaking abounds, etc. Rulers are the best! 

Until, one day, you discover a series of objects _significantly smaller_ than your unit of measurement. You can't construct rulers small enough to make a new unit of measurement. Worse, you can barely percieve them with your naked senses. Measuring length is dead! Measuring is dead! Empiricism is dead! 

Except, that's not what happened. We measure small things all the time. When we needed more accurate measurements, we invented new tools for measuring them. We stuck a bunch of lenses in a tube and made a microscope.

# New Evals

"Evaluations" are just our way of getting a measure of how models perform at certain tasks. There are "rules", but those rules are pretty changeable!

Not long ago now, we used to care a lot about [MMLU](https://huggingface.co/datasets/cais/mmlu). We liked it because it:

1. Easy to develop in parallel: Dan Hendrycks and other researchers could work on gathering the problems without necessarily needing to collaborate directly. 
2. Measured something we were interested in - how many facts does an LLM know?
3. It was easy to grade. Multiple choice helped us out with how sketchy LLMs were at outputting structured objects.
4. It was time to replace stuff like [GLUE](https://arxiv.org/abs/1804.07461), which was essentially measuring nothing. 


That is, MMLU was an evaluation that was designed to function with the tools we had available for measuring. In some sense, the _lowest amount of effort_ that was sufficient to create the instrument for measuring, directionally, what we were interested in about models. 

We then moved to a world where we measure _agents_. By default, we put those in a tool-calling harness, set it up facing a docker container representing an environment, and we have some kind of python function that checks whether what we wanted to have happen did in fact happen.

That was great! It was also super parallelizable, we could get good abstractions around it, and it was "regular" enough that with some examples you could steer a model into making more of them, meaning you could mass produce evaluations for things you knew a lot about with a mostly alright amount of reward hacking. That's how we got SWE Bench! That's how we got LiveCodeBench! 

When the rulers not enough, we invent a magnifying glass. When the magnifying glass isn't enough, we invent the microscope. 

We live in a world where the models are capable enough to [solve navier–stokes Millennium Prize Problem](https://openai.com/index/navier-stokes-solution/) and [collude on evals by hacking production infrastructure](https://www.youtube.com/watch?v=87DyyMV0kCY) and we are still mostly testing them on small docker environments and programatically defined win conditions. Are the models not capable of building better instruments to measure their capabilities? Or are we just lacking in our ambition to try and measure those things?

In no particular order, some particular evals or eval-techniques that I'm excited about. 

## Terminal-Bench

The 🐐.

### Harness as first class

From a pure empirics perspective, I'm excited about Terminal-Bench because it treats the harness itself as part of the measurement tuples of `<model, harness, success %>`. In many benchmarks the model just sits entirely alone. Obviously that's not the way things get run. Terminal-Bench represents that reality accurately!

{{< figure src="glm-claude.png" alt="" caption="That's 30pp. We're not quibbling over noise here." >}}

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

Violently whittling down over 900 tasks proposals to 70 tasks that are high quality enough to launch your _0.1_. This isn't done by accident, you can check out the [paper](https://arxiv.org/abs/2601.11868) to get a sense of how they construct such a high quality benchmark. While Tbench is the benchmark that is most "recognizable" out of all the benchmarks on this list (mechanistic verifiability, one-shot tasks) its commitment to quality and its hoisting of the harness to first-class attribute make it laudable to me. 

## (Agentic) Judge Evals

If you've met me in real life, you can probably skip this section. I'm always talking about this. 

## User Sim Evals

## Swarm Evals

## Monitor Evals


[^1]: If you're interested, you can see them [here](https://aitp-conference.org/2026/).

[^2]: Relative to scrolling twitter or looking at a PDF. It's not, like, Millenium prize problem hard.  