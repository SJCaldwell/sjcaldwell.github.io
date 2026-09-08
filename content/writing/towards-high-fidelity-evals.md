---
title: "Evals Aren't Dead: They Need to Grow Up"
date: 2025-08-17T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "evals"]
tags: ["llms", "evals", "research"]
description: "It's not 'evals' that are dead, but there's definitely a subset of easy ones that are dying."
summary: "It's not 'evals' that are dead, but there's definitely a subset of easy ones that are dying."
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

## Agentic Judge Evals

## User Sim Evals

## Swarm Evals

## Monitor Evals


[^1]: If you're interested, you can see them [here](https://aitp-conference.org/2026/).
