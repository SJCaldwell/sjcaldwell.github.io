---
title: "GPT-5 is Good, Actually: The Agony and Ecstasy of Public Benchmarks"
date: 2025-08-17T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "evals"]
tags: ["llms", "evals"]
description: "An attempt to explain why benchmarks are either bad or secret, and why the bar charts don't matter so much."
summary: "An attempt to explain why benchmarks are either bad or secret, and why the bar charts don't matter so much."
ShowToc: true
TocOpen: false
draft: false
---

My first reaction to GPT-5 was positive. I was at Blackhat/Defcon the week of the release, and was mostly (for once) not on Twitter. Being off Twitter, I missed the much maligned livestream. In fact, before I even got on Twitter and saw everyone absolutely clowning *that* bar chart, the first things I heard were positive.

My younger brother works as an accountant. Living mostly in Excel and manually reconciling data between a lot of systems, he doesn't use the models all that much. In our group chat, he posted a screenshot of his first GPT-5 interaction. It was an accounting question about how land depreciates (it doesn't). GPT-5 was the first model that got his question correct. He said, basically, "Maybe they finally got me[^1]." Some other friends who work in data science and infrastructure also basically complimented the model. Some of them pay for the highest tier of GPT access, others pay for the lower subscriptions. Nobody was totally blown away, but the general reaction was "impressed in the way I expected to be impressed". 

Then, I logged on to Twitter. 

{{< x id="1953504619969306981" user="_xjdr" >}}

There was a lot of deserved haranguing about the chart crimes. It's a bad chart. It's a chart so bad it's difficult to imagine in a high school science fair, much less the much hyped release of the most significant lab on the planet. The implication was that the chart needed to be bad because it wouldn't look good if you displayed it accurately. GPT-5 without thinking is worse than o3, and the total gains of 5 with thinking are a measly 6%. AGI is cancelled, everyone pushed their timelines back, RL has no more gains to give, etc[^2]. 

So, who's right? My brother, with his one question vibe check or 80% of Twitter, with their ability to competently read benchmark bar charts and from bar charts tea leaves?

I'm going with my brother on this one. GPT-5 is a great model. If o3 had never been released, people would be losing their minds. But it does exist, and so the result is iterative. But being iterative doesn't prevent it from being _very good_. And as the week went on, some smart people I respect found the model generally more capable than its predecessors/competition.

{{< x id="1956423766768247032" user="moyix" >}}

{{< x id="1955286758700056927" user="_xjdr" >}}

I've found generally the same things. GPT-5 has been super useful as a research assistant in the last week. Its ability to find relevant paper results for lit review has increased dramatically, as well as its ability to proofread paper drafts and work through technical specs. I haven't used it for code, yet, but I'm so happy with Opus I just haven't bothered. I'm confident it's quite good at that, too.

So if it's so much better at code re: xjdr's tweet, how come that doesn't show up in the SWE Bench results? That's easy, public benchmarks are basically awful and you're better off ignoring them.

## Public Benchmarks are Terrible

I'm going to say a lot of harsh things, but before I do, I have to acknowledge: 

{{< figure src="evals_suck.jpeg" alt="" caption="Too real." >}}

Evals are incredibly difficult to make. They're getting more difficult every year. The people who manage to do it are undersung heroes and nothing I write here is to criticize them. If you want to understand why it's so difficult, there are two salient points to understand:

1. As soon as you make a benchmark public, it is going to get totally saturated if anybody cares about it, and then it might as well not exist.

2. The models are so capable that creating an evaluation capable of distinguishing between the most capable models is expensive and painful.

Before we expand on those, let's just briefly talk about the good old days with training sets and test sets. 

### The Good Old Days

The ideal benchmark dataset is difficult enough that substantive progress on it requires serious breakthroughs. ImageNet, for example, was a large and broad enough dataset such that doing classification well required the creation of convolutional neural nets. When researchers refer to the [ImageNet Moment](https://image-net.org/challenges/LSVRC/2012/) they're referring to the 2012 rendition of the ImageNet classification challenge where [AlexNet](https://en.wikipedia.org/wiki/AlexNet) won the competition with over a 10% lead to all of its competitors, and would spawn 80,000 citations and a whole slew of technical innovation in the years to follow. ImageNet itself was created in 2009. That's four years! SWE-bench Verified came out [last year](https://openai.com/index/introducing-swe-bench-verified/) and it's cooked.

The rules were also very clear. Everybody had the same training data. The test set for everyone was the same. If you trained on test this was immediately clear from trying to replicate your results, and if you did that you would be sent to the gulag. You could look at both sets and have a sense of what generalization was required to perform the task, and when a method "worked" it was obvious to everybody. That's no longer the case.

Now the training set is R E D A C T E D. We have no idea what frontier labs are training on, but it's as much as they can get, then as much as they can generate, and then as many worthwhile environments as they can get verifiable rewards from[^3]. There's pretraining, mid-training, post-training, with different teams working on different parts of the training. Let's take a look at everything the [GPT-5 model card](https://cdn.openai.com/pdf/8124a3ce-ab78-4f06-96eb-49ea29ffb52f/gpt5-system-card-aug7.pdf) has to say about the data and training. 

{{< figure src="gpt5_model_training.png" alt="" caption="Thank god it all fits in a screenshot." >}}


That's nothing! You're actually better off hanging around their [careers page](https://openai.com/careers/) to try and get a sense of what capabilities they're trying to bring to the team (and models). And OpenAI is in no way special in this, that's just how the labs are these days. Every piece of information is a freebie to a competitor and they've got enough to worry about with the way information flows around SF. Beyond that, every written admission of how anything was trained invites a potential legal challenge. It just doesn't make sense to say anything. If you want a sense of what data is being used to train a model, you can stick to [Allen](https://allenai.org/) and [Nous](https://nousresearch.com/), but even the leaders of those labs would agree that they're far more resource bound than their frontier competitors and their models lag accordingly. 

So the training set is ???, the test sets are these public benchmarks/evals, and the test-time distribution we'd like these models to cover is _literally anything you might want a computer to do_.  

With that established, let's cover those two points from earlier:

### Public Benchmarks Will Always Be Saturated

The preprint of [SWE-Bench](https://arxiv.org/pdf/2310.06770v1) was released in October of 2023. The creators took 2,294 public issues from 12 popular Python repos. These include astropy, seaborn, pytest, flask, sphinx, requests, pytest, and others. These issues and models performance on them have essentially become the single scalar of how models are perceived at performing on software engineering. 

This is an ingenious idea for a benchmark. You've got all this code data out there, and the creators had an intuition that writing one-off functions to get specific test cases to pass was missing some of the complexity of real software engineering and that these public Github issues of mature projects presented a really useful measurement of progress. They set up a harness to test models and report that the _best_ model earns a 4.8% on their benchmark. That seems really great, and like it's going to be useful to watch models slowly improve at it, and as they improve on these benchmarks we'll see gradually better coding capabilities in the models. 

But that's not really what happened. By publishing this benchmark and it becoming the de facto measurement of model quality for what is currently the most economically valuable task LLMs can work on, it became the battleground for frontier labs to fight it out over[^4].

The ImageNet of it all falls apart almost immediately due to the incentive structures at play. Training a model is super expensive, nobody gets to see your training data, and most people who aren't using these models at a high level are going to judge you mostly on this score. Even if the models were trained exclusively by saints, it's not hard to figure out what's going to happen. You can be damn sure that as they're training these models they're taking a look at the SWE-Bench leaderboards and figuring out if there's a narrative where they're a helluva lot better, or very competitive for the model size, or _whatever it has to be_, but there has to be a narrative that looks good or that model isn't going out the door.

Train on more code? Sure. Set up RL environments that are shockingly similar to the benchmark but using different repositories? Literally why wouldn't you? Your competitors are. The delta between evals and RL environments all comes down to whether you're willing to write a reward function and update some weights. Schemes to generate synthetic data that is intentionally close to the test set but _isn't_ (legally) the test set? Please do. 

The fear of training on the test set previously was that your model would memorize it all and then totally fail to generalize at all to the real world. Now that's not nearly so much of a concern, you can do whatever black magic you need to in order to get the numbers where they need to be, and that's just another item on your to-do list as you prepare for a major model release. That doesn't mean you're making a bad model - I've personally seen the capabilities of the models continue to increase at a steady rate that continues to blow my mind. It's just that also you make sure you count the letters in strawberry correctly because you know that's something people are looking for and you're tasked with brand building at the same time you're tasked with creating the most useful model possible.

Then, having gotten the model as good as it's going to get, it's time to dress up those results. Need to mess with pass@k for its bar on the chart to be taller than the other guy? Fine. Need to beat a Y-axis to death with your bare hands such that it violates Euclidean geometry? Cost of doing business. Nothing about it is really surprising. You've all worked at places where somebody made a slide deck about your work that hyped it up more than is deserved, and if you've lived long enough you've come to accept that that's just one of the weird perverse incentives of business. Epistemically fraught, a bit, but if everybody's in on the game it's not shocking or anything. It's just what it is.

So as a researcher without access to a frontier labs compute, the most useful way you can steer the lever of progress is by developing large, easy-to-run benchmarks that models are currently kind of bad at for tasks you care about. This is an incredible amount of work in itself. Backbreaking amounts of quality control, one-offs to fix, and mental labor expended. If you then do the work of getting that benchmark popular and well-cited enough, it goes into the crosshairs of the labs. If your benchmark comes to matter enough to be referenced in the model card, it's going to get saturated[^5], because these labs have to one up each other every time a release comes out, so you are nearly guaranteeing that those capabilities are going to increase, but also that the benchmark isn't going to matter much anymore. Or at least, the climbing of the benchmark numbers are not going to be as aligned with the capability increases you see in real life as you hoped there would be when you made the benchmark.

I mean, Jesus, even playing Pokemon got saturated.

{{< x id="1955980772575268897" user="Clad3815" >}}

So, what do you do? You accept the Dark Forest situation for evals and work from there. You keep secret benchmarks that aren't available to frontier labs and in that way you have your own private signal of model capability increases. The downsides of this are it's still really hard. Benchmarks are _not_ easy to build. Creating a set of reproducible, diverse tasks that are complex enough to be worth keeping track of is just an inherently difficult thing to do. But if you get it, it's my little brother's accounting question on steroids. Crucially, this makes _no_ sense if you're a researcher. Releasing a really strong benchmark is a ticket to fame, fortune, and maybe some of that compute you currently don't have any of. So who does this make sense for? Businesses, governments, the types of organizations where people would find it worth investing in understanding capabilities and then keeping that knowledge to themselves.

What are the epistemic downsides? Well, let's see what happens when you tell somebody about your definitely very real and intentionally secret benchmark.

{{< x id="1956686066708070816" user="MrTuxracer" >}}

I get it, Mr. Tux, I really do. But if they let you verify those benchmarks (made them public and verifiable), they would lose all meaning almost immediately. How do you know how to update your beliefs based on a company's report of a benchmark if you can't verify it? Well, depends on your belief of the integrity of the company. So we arrive at the _just trust me bro_ era of AI research. Blessed be the Twitter hypebeasts who show off their cool examples on Twitter, because if not for them you'd have no signals at all. This is why people who use LLMs in some vertical release cool demos and try to put out public examples of their work. They have to find some way to send you and other potential customers positive signal that can combat your basic skepticism over claimed capabilities without just releasing their benchmarks and making the entire exercise pointless. 

### The Models Are So Capable They're Hard to Evaluate

Evals are hard! They were hard "back in the day" and they're harder now. MMLU seems like a relative cakewalk from an infrastructure perspective. If you can put out your _whole_ benchmark on HuggingFace and it all works by downloading a dataset and running it you have it as easy as possible. The quality control required to make several thousand test cases all correct is still extremely painful and labor intensive, but at least it's easy to run. 

But we don't _care_ about question answering now. Or translation. We care about stuff like computer-use. Now that we're evaluating agents, each of these tasks needs realistic and rich environments. Someone has to make that! That's a lot of engineering, expensive infrastructure, and domain expertise to make sure you're not fooling yourself. When orchestrating 500 docker containers is the clean case, you know it's going to be painful. 

As these setups are required to get more painful in order to accurately measure the capabilities, they're also just more expensive to run. The infrastructure needs spinning up, the token use to get to an action turn count such that you can prove your environment is sufficiently realistic and the task is sufficiently difficult is huge. Trust me, pal, you wouldn't run those evals if they _were_ publicly verifiable. You don't have the stamina or the checkbook. 

That in and of itself is one of the largest markers of progress to me. It is legitimately an intellectual exercise and engineering undertaking to get a truly useful set of scenarios where the models actually screw up. That was not the case in 2023. A lot of smart people are spending a lot of time trying to get to an empirical measurement they can trust for their particular domain. And that ability to _measure capability_ in and of itself now becomes intellectual property, and it's pretty likely those who invest the effort are going to keep it to themselves.

### What This Means For You

So do I have any actual recommendations here? Sure, build your own benchmarks. If you're an organization, this is basically a must. It's hard and requires a lot of effort but if you've got a business case around models reaching a certain capability level, it's basically table stakes to be able to measure those in a mature and repeatable format. Nobody wants to write evals, nobody wants to run evals, but if you're not participating you're left looking at benchmark screenshots. This is, essentially, irresponsible and ensures that when the capabilities get to that point you were waiting for you'll find out about them via tweet.

If you're an individual? Well, the least you can do is get your private test set together. This could be questions, this could be engineering requests or code you'd like to see, it could be a harness you expect to be able to accomplish some challenge agentically when the models get good enough. You don't have to tell anybody about it, but you should have them. They'll tell you more than the bar charts of those publicly available evals you've never examined. And you'll be able to comfortably skip the livestream and decide for yourself if GPT-6 is any good.

[^1]: His language was a bit more severe and quite a bit funnier, but it's bad form to directly quote Signal GCs. 

[^2]: In fairness this probably also has a lot to do with the model routing, which was apparently [broken](https://x.com/tszzl/status/1954325217087754278) on day one. 

[^3]: And then whatever the universal verifier (judge) tells them is good, and so on, and so on.

[^4]: And usurped MMLU as the bar chart people look at before they tweet whether the model is good or not.

[^5]: The only area where this is spiky in my personal experiences is cybersecurity evals, where the incentives seem to shift to desiring to look non-threatening and not worth legislating. Sometimes I look at results on stuff I run and the output of frontier labs and assume they're tying the model's hands behind their back and leaving them a python 2 interpreter, bash, and some duct-tape so they can report the models are still only kind-of-okay at CTFs. Trust me, they're really quite remarkable.