---
title: "ProofJudge: Can we align vibe-proving with human taste?"
date: 2026-03-29T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "evals"]
tags: ["llms", "evals"]
description: "Towards measuring alignment with human taste in autoformalization with judge agents."
summary: "Towards measuring alignment with human taste in autoformalization with judge agents."
ShowToc: true
TocOpen: false
draft: false
---

>I think in the future, there will be entire professions of mathematicians who might take a giant Lean-generated proof and do some ablation on it, trying to remove parts of it and find more elegant ways. They might get other AIs to do some reinforcement learning to make the proof more elegant, and **maybe other AIs will grade whether this proof looks better or not**.
>
> -[Terence Tao](https://www.dwarkesh.com/p/terence-tao)

Today I'm releasing ProofJudge - an eval I built for testing LLM-as-judge alignment with expert human taste on Lean4 code quality. The release includes:

- [The harness for ProofJudge](https://github.com/SJCaldwell/ProofJudge).

- [The evaluation dataset for ProofJudge](https://huggingface.co/datasets/SJCaldwell/proofjudge).

- [A dataset of traces for the judge agents being run against ProofJudge](https://huggingface.co/datasets/SJCaldwell/proofjudge-eval-traces): These are in [ATIF](https://github.com/harbor-framework/harbor/blob/main/docs/rfcs/0001-trajectory-format.md) format. 

## Why It Exists

While I was writing [The Tests All Pass](https://hackbot.dad/writing/tests-all-pass/), I ended up thinking a lot about the lack of verifiable _taste_. The last few months have been a rush of engineers trying to figure out how to get agents to write software, and a lot of people being burned by that software not being very high quality. We might say that code is optimized to be able to _run_, but not optimized for being able to be _read_, _re-used_, or _changed_. 

Escaping to somewhere taste didn't matter sounded ideal, and provably correct software seemed like the place for that. My assumption was that formal verification made taste irrelevant - if it compiles, it's correct, and you're done.

[Mathlib](https://github.com/leanprover-community/mathlib4) is the flagship formalized mathematics library for Lean. I expected it would be a huge flurry of activity with tons of AI PRs getting merged as everyone threw compute at their agents on the great quest to "formalize everything". I was right about the former, and wrong about the latter. Oh, it's not that people weren't _trying_ to throw their agents at random proofs and getting them into a PR, it's just that they were all soundly rejected as slop. I saw dozens of PRs closed for pretty much the same reason. 

As it turns out, proving something is correct is _not_ where the bar is for having a PR accepted into Mathlib. There's an entire [bootcamp](https://yaeldillies.github.io/mrb2026) where they work hard to make sure folks are well prepared to be mathlib reviewers. The LLM proofs tended to be unnecessarily long, awkward, with no re-usable pieces pulled out such that they were useful to others. So they were definitely correct, but they weren't _useful_ to people using mathlib to prove new mathematics, and they weren't useful for anyone reading them as a guide to learning why something was true. So they were _technically_ correct, and practically useless.

In fact, there's been tons of drama in the Lean community about AI companies in the space for that very reason. Basically they're being accused of getting credit for these massive autoformalizations that are now technically proved but offer no value to the community. Because they've now been "proved", a human trying to make a high quality (say, mathlib-ready) version of the same code will get very little credit and have little motivation to do the work the right way, because they won't get credit for the value being created.

{{< x id="2035827665869947226" user="ludwigABAP" >}}

The most recent high-profile example of this tension peaked when Math Inc received tons of press for their [sphere packing formalization](https://github.com/math-inc/Sphere-Packing-Lean), with 500k lines of Lean code.   

That's not to say there's not interest in AI formalization as a tool, but as far as I can tell from lurking Zulip, in general a generative proof is then tastefully cleaned up by a human being before ultimately being merged. There's a lot of subjective taste being applied before that happens, and for those without it, the PR simply isn't being merged.

The tests all pass, nobody wants the PR. 

If we wanted to create a system such that you could RL on taste, and make your solution correct _and_ high quality, you would want an ML system that was capable of judging that quality. Figuring out how they perform on that was the motivation for building out this eval. 

## What It Is

Each data point is a PR into [mathlib](https://github.com/leanprover-community/mathlib4) - the initial version (that was not merged) and the final version after requested changes were made (that was merged). A typical example might look like a reviewer asking for an intermediate lemma to be pulled out of a monolithic tactic proof, so the result can be reviewed by other users of the library.

The LLM judge is presented each independently and given tools to explore the PR and its place in the codebase, before rating the PR from 1-7. In order to be considered _aligned_ with human feedback, its ratings for the final PR must be higher than the initial PR. If it rates the earlier version higher, or scores them equally, that is considered a failure.

I've got 123 cases here across 100 PRs. To help judge, the LLM is provided tools that allow it to read the code in the PR as well as the surrounding mathlib code at the time the PR was created. This is to ensure nothing is being implemented redundantly, and to give the judge a chance to look at the mathlib code assuming it hasn't been trained on a lot of it, such that it can better understand whether this code "fits". It is _not_ at all concerned with correctness of the proof.

Mathlib has a few quirks about its PR process that made this data collection complicated. Force pushing to PR branches, or draft branches that are later opened brand new once some review has been done, mean often the differences between the initial rejected code and the final accepted code are subtle. Since this just makes the eval harder, I decided that was an acceptable limitation. In [previous work](https://arxiv.org/abs/2508.02921) I've found getting human baselines for judgement to be tedious and difficult. It would be best if I had it for this eval as well, but I don't (at least for this version). My argument here is that by construction, there are clearly human preferences being expressed over what is good enough and not good enough to be merged into mathlib. Obviously there's not a singular reviewer of mathlib, and so there's some bias creeping into the dataset that way. 


## How Models Perform

Anything above 3/7 or ~43% is beating random judgement. 

| Model | Provider | Alignment | Cost/Pair |
|-------|----------|-----------|-----------|
| gpt-5.4 (reasoning: low) | OpenAI | **65.9%** | $0.24 |
| Kimi K2.5 | MoonshotAI | **59.3%** | $0.07 |
| Gemini 3 Flash | Google | **58.5%** | $0.22 |
| Qwen3-32b | Groq | **52.8%** | $0.013 |
| gpt-5.4-nano | OpenAI | 50.4% | $0.009 |
| gpt-5.4-mini | OpenAI | 48.8% | $0.03 |
| Haiku 4.5 | Anthropic | 46.3% | $0.56 |
| gpt-oss-120b | Groq | 27.6% | $0.014 |
| Gemini 3.1 Flash Lite | Google | 26.8% | ~$0.001 |
| gpt-oss-20b | Groq | 5.7% | $0.018 |

With judges, I like to look at the pareto frontier so you can get a sense of the different cost-levels. 

{{< figure src="proofjudge_pareto.png" alt="" caption="Pareto frontier of Judge Agent performance" >}}

Kimi continues to dominate on a per-cost basis, coming impressively close to closed lab performance with open weights, which was also true of the evals I ran for [PentestJudge](https://arxiv.org/abs/2508.02921). There's no better model to distill from at that cost point. I'm pleased to see Qwen3-32b sitting reasonably above random, since I think it would be a great candidate for finetuning here with modest computational resources.

Failures, even for frontier models, tend to look like ties. That is, following the rubric led them to provide a score that neutral (5) for both PRs, leading to a tie failure. Gemini Flash Lite assigned literally every single PR with the same score.

That remains true even for the models that do the best. The failures are 27.6% ties and 6.5% inversions for GPT 5.4, so if the ties could be meaningfully broken, we would have an extremely strong judge. This may well come down more to rubric design than actual limits of the models. If you're a Mathlib reviewer and you think the rubric is wrong, that might boost us by almost 30% accuracy.  

Obviously there are some big models not tested. Sonnet, Opus, Gemini 3.1 Pro, or GPT 5.4 with higher reasoning. In my defense: they would've cost me a couple hundred bucks. There aren't standard errors on these bars for the same reason - each model was run only once. Because the results are stochastic, it's hard to say presently how consistent any of the models are.

If anyone wants to run those models in the harness, or give me some credits, I'll be happy to update them! 

## What's Next

This evaluation is in its early stages: I'll be submitting these artifacts along with a paper to [Artificial Intelligence and Theorem Proving](https://aitp-conference.org/2026/). I hope to use this as an opportunity to build out the most useful version of the evaluation possible along with the community, with the ultimate goal of making a small and cheap judge that could be useful for reviewing proofs, and ultimately make it easy to do RL against a goal of making a PR not only verifiably correct, but with taste likely to make it accepted by Mathlib, and therefore useful to the math community broadly. In particular, if you're a mathlib reviewer and can help me build out a stronger evaluation set and more useful [rubric](https://github.com/SJCaldwell/ProofJudge/blob/main/rubrics/rubric.md), I hope you get in touch. 