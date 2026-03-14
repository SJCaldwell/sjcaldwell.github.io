---
title: "The Tests All Pass"
date: 2026-03-14T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "evals"]
tags: ["llms", "evals"]
description: "METR's SWE-bench analysis shows us taste isn't verifiable."
summary: "METR's SWE-bench analysis shows us taste isn't verifiable."
ShowToc: true
TocOpen: false
draft: false
---

This week, METR released [Research note: Many SWE-bench-Passing PRs Would Not Be Merged into Main](https://metr.org/notes/2026-03-10-many-swe-bench-passing-prs-would-not-be-merged-into-main/). It has me more confident than ever in the necessity of llm-as-judge research for evaluation and post-training. 

What they attempt to capture is what percentage of SWE-bench patches that pass the autonomous grader wouldn't be accepted in a real PR. It's one of those impressive pieces of LLM evaluation where they manage to get human beings involved, albeit a few of them, to review submitted SWE-bench patches to determine whether or not they would be merged. In addition, maintainers were (blindly) asked to review the golden patches - that is, the actual patch the SWE-Bench challenge was based on. Assuming the maintainers rejected these patches, they were asked to clarify why. Those reasons being:

**Code quality**: Bad style, not up to standard of the repo.

**Breaks Other Code**: Solves the issue but does it in such a way that it breaks other parts of the software.

**Core functionality failure**: The actual problem is not satisfyingly addressed.

That established, let's take a look at the results of their analysis:

{{< figure src="swe_bench_rejections.png" alt="" caption="" >}}

Things are improving, as anyone who has been using these models seriously can tell you. 

However, it's obvious an automated grader's "pass" in SWE Bench Verified is _correlated_ with improved professional SWE coding ability, but not cleanly lined up with it, in a way that affects both are ability to measure capabilities and our ability to train for them.

[This is not a particularly naive grader, either](https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/grading.py).  Basically, in order to get a pass from the automated grader, two conditions have to be true. "Fail-to-pass tests" fail before the PR was merged, but pass after. These are typically the tests that were added as part of the PR. The hope is these define that core functionality. Then there are "pass-to-pass" tests, which are tests that completed successfully before the patch. If introducing the patch breaks them, then the agent broke other code. Sounds very reasonable, and someone clearly thought about edge cases as the dataset was being gathered. 

Immediately, the fact that these two categories _exist at all_ are showing a failure of the metric being measured. Given that SWE Bench was initially created in a semi-autonomous way, that may not be surprising, but note that [SWE Bench Verified was already filtered with human review to ensure that the issues were correctly specified and the tests were relevant](https://openai.com/index/introducing-swe-bench-verified/). Despite that additional effort, it's still possible for a passing patch to break parts of the functionality of the application that the pass-to-pass tests do not capture and can be seen by human review, and create something that creates a passing unit test but does not actually resolve the intended behavior in the issue request. 

Pass-to-pass tests are insufficient - this is unsurprising. A lot of software has parts that aren't tested, or are maybe unit-tested but not tested end-to-end. We have all written code that the tests in our CI would pass that nonetheless would break some functionality of the application[^1]. If we wanted to avoid this failure mode in post-training, we add additional unit tests, or more thorough end-to-end tests. This will make the verification step slightly slower, but is tractable to do at scale with coding agents. Raise the bar required for reward hacking. 

The fail-to-pass is interesting because it's related to a behavior many of us saw firsthand with coding agents last year, and we can connect it more explicitly to post-training ideas. The agents were willing to write _bad_ code so long as the resultant code ran. For example, if it couldn't connect to an API, it might mock out some sample data it would receive from that data if it _could_ connect to it right in the middle of the function intended to call said API. It smelled very *reward-hacky*, like this is the kind of software an agent just being graded on whether functions passed or not might write. In this case, someone doing TDD would have this same experience: specifying behavior and finding software that technically fulfilled the requirements while not doing the difficult thing you wanted it to do. 

This can be resolved, in some ways, by more tests. If your addition function that checks that 2+2=4 is being hacked, parameterize it and test more cases. Eventually it'll be easier to just write the function you're supposed to write. Even better, you could hide the exact tests being run from the agent, and simply report a failed input/output when a submission comes along, so reward hacking is more difficult. This is how LeetGPU and Leetcode do it. You get a few sample cases, but there are plenty of input/output pairs you won't see unless you hit that particular case. This encourages the general solution.

This is slightly more painful from a scaling perspective because it's more complex, but still very tractable and we might expect agents to be better at resolving that core functionality as defenses against reward-hacking become more capable.

If you're purely interested in verifiable rewards, it is plausible to think of solutions to the "core functionality" rejection category and the "breaks other code" rejection category. What about code quality?

## What's Slop?

How would you write a function that measured code-quality? Imagine you wanted to take the verifiable signals that SWE-Bench Verified has for grading and use them as a training environment. Based on this METR report, you decide to use coding agents to augment the verifiers so they include end-to-end tests for the pass-to-pass portion, and more thoroughly test the feature and hide the inputs from the agent for the fail-to-pass tests. You model improves, and has some of its reward hacking beaten out of it. Nothing breaks in the software, and the core functionality works just fine.

Now you have one problem left, the maintainers still don't want to merge the code because it's _gross_. 

I'm not a software aesthete. I have always admired people who were "cathedral builders" but I'm more of a "software is a box that solves problems" guy[^2]. Still, agents often create code that is simply too ugly for me to deal with.

Most frequently I find I can dangerously skip permissions and keep prompting for the majority of the early phase of a project. I tend to know how _I_ would solve this problem, so can make suggestions if I see something that looks dumb. The output looks good, and I am generally happy so long as I stay mentally involved with inputs and outputs. If I'm lucky, a project ends at this stage. There's so many one-off analysis scripts, bespoke TUIs, and dataset scrapers that simply stop being developed as soon as I've accomplished my task, and I leave having saved a few hours and being a happy man. 

Should the project progress, however, I will hit some issue that I feel the agent is just not getting any closer to solving. It might _feel_ very minor to me, but the agent is just having trouble understanding what I'm getting at. Now I have to jump into the code. No problem, I ask the agent to give me a tour, it highlights the major functions I need to familiarize myself with, and I hop in the editor. 

Folks have described feeling like they're losing the ability to read code due to over-reliance on agents. I don't know if I agree with that in a meaningful sense, I think we're a bit too early to be shedding those skills, but I certainly have trouble reading _agent_ code. And it's usually because the agent code simply has no taste for abstractions whatsoever, and isn't keen on cleaning up after itself.

The example that sticks out most cleanly is what it is willing to do from a data transformation perspective. Data classes, pydantic models, and untyped dicts abound, freely being casted between each other in-between business logic, all highlighted with comments that explain the motivation behind this particular transformation. It all looks extremely reasonable, but as you try to build an inventory of what's-doing-what, it all tends to blend together. It _works_, mostly. It _runs_, mostly. But you have trouble reading it and the agent has trouble logically pulling it all into context with its search agent such that it can perform in it effectively. [I know it when I see it](https://en.wikipedia.org/wiki/I_know_it_when_I_see_it)[^3]. How do I define a function for a definition that informal? 

## Learning the Slop Detector

The question of "how would I write a function to detect slop" is a question very similar to "how would I write a function that detects birds". You simply are not going to arrive at a formal bullet-proof coded up scheme for detecting something like that. Humanity settled on using learned functions for that sort of thing and it was extremely effective. Here we have arrived back at RLAIF/RLHF from first principles. It's hard to write a function for whether a response is "helpful, honest, and harmless". You would know it if you saw it, and it should be easy to provide a _rubric_ that a competent LLM could use in order to evaluate the code. 

Judges obviously create certain complexities from an evaluation and reward perspective. They are stochastic in nature, so your judge will need its own evaluation in order to get an estimate of how much error is baked in. It costs more compute to run, and setting up the infrastructure for training and evaluation has an extra piece of work when you're lugging around a whole other model. 

These are real challenges, but ultimately, they're just engineering problems. Let's consider an alternative: fleeing to domains easier to verify. 

## The Limits of Verification

The last few months have seen software engineers attempt to figure out precisely how to use these coding agents most effectively. On the maximalist side, you have engineers attempting to make Factorio out of it. There's _no reason to read the code_ as long as is it runs and does what you want it to do. If humans don't read it, it's cheap to run, and it works, just focus on a higher level of abstraction. All the help you really want to give to this process is verifiability to shape the output - do these tests pass, do these precommit hooks pass, do my end to end tests work. Outside of that, _don't care_.

If you take this idea seriously, you end up wanting more out of your programming language than unit tests. Despite our best efforts, we're responsible for the engineering artifacts from these coding processes even if we haven't read the code. You would want to be in a situation where a human being was responsible for a _spec_ and a language guaranteed that spec was implemented correctly, allowing the codebases to grow like gnarly biological organisms without causing any strife to the human who "owns" the codebase or requiring considerations that weren't hard verifiable. 

Thinking along those lines, I've been investigating Lean4 for proof-writing. It's as verifiable as it gets, and mathematical statements can create a perfect spec. My naive view was "as long as this result is proved to be true, it's good, and who cares how?". Your anticipation, then, would be that huge branches of mathematics were being autoformalized _right now_ and human beings were just making sure the "specs" looked right.

[Mathlib4](https://github.com/leanprover-community/mathlib4) is a library designed to essentially include a bunch of useful results from mathematics that are already formalized that you can use to formalize whatever new math you're trying to prove. If you're in the middle of something thorny you don't want to take the time to get a good proof for `a + b = b + a`, you want to use the one that already exists. Lean4 is built on a relatively small body of axioms, so there are many things you would have to prove yourself without this kind of library that would be very annoying and wasteful. It's extremely active and has new code merged daily.

Many of these PRs turned out to be ai-assisted, but nearly none were written completely by AI. Most of them required human oversight. I read dozens of issues where someone claimed to be formalizing tons of mathematics at a time using `gpt-5.x-reasoning-BIG`, and Lean4 was declaring all the code correct, and yet the code was being rejected. What gives?

As it turns out, code _quality_. What might be non-intuitive to those of us who don't use Lean frequently is that a library of mathematics is only useful if it features an elusive and taste-driven compositional reusability. For any results you've formalized, it is important to determine which lemmas are general enough to be pulled out and set up as a foothold for future proofs. The PR conversations tend to be _all_ about that, and Lean4's verifiability has nothing to say about it. The unit tests all pass but the maintainers won't merge. The taste isn't formalized, and it isn't verified. A judge still sounds very attractive here for evaluation in this verified landscape, and [my own research](https://arxiv.org/abs/2508.02921) suggests it's very tractable.

What seems more speculative is whether it would work for training for complex tasks. Luckily, we don't have to speculate.

## Judges as Reward Signals

[QED-Nano](https://huggingface.co/spaces/lm-provers/qed-nano-blogpost#introducing-qed-nano-a-4b-model-for-olympiad-level-proofs) from LM Provers provided a really nice testbed for this judge based training. Essentially it was being used to grade informal mathematics proofs (the proofs are still right or wrong, but they're not written in Lean) of a model during training. GPT OSS 20B is used as the grader of the rubric, and the learning signal is dense enough that a 4B parameter model gets within spitting distance of Gemini 3 on mathematics evaluations.

The [grading criteria](https://huggingface.co/spaces/lm-provers/qed-nano-blogpost#setup-training-prompts-and-grading-schemes) are created _per problem_ so they're very specific. This actually sounds like _stronger_ evidence of the efficacy of judges to me. GPT-OSS-20B was obviously not trained for this task and has seen relatively few examples of any of these particular problems being solved, and yet a very specific rubric performs well enough to provide a reward signal for long running RL. The authors note they found _no evidence_ of reward hacking, and the human grader and the llm judge agreed on most problems. 

This does not mean it will be easy and tractable to train models that understand slop, nor does it mean that we should throw away the benefits we've had from strict verifiable rewards. The natural next question is what happens when you combined a formal verifier for that parts that are checkable and a judge for the qualities that aren't to nudge the solution space towards artifacts we're happy with.

[^1]: Even very well tested applications do not simply accept PRs so long as all the tests pass!

[^2]: One imagines I was exposed to Python and data science too early in my education to end up as anything else.

[^3]: I fully confess this is an admission of a skill issue. I have seen software that looks [quite nice](https://github.com/Noumena-Network/nmoe) that seems to be mostly if not entirely written by coding agents, it just seems to require running those agents like the navy in a way I haven't worked out yet and certainly requires a human being to remain locked in.
