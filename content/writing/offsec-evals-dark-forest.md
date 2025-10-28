---
title: "Offsec Evals: Growing Up In The Dark Forest"
date: 2025-10-28T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "evals", "offsec"]
tags: ["llms", "evals", "offsec"]
description: "If you contribute a public benchmark, are you giving free capability to your competitors?"
summary: "If you contribute a public benchmark, are you giving free capability to your competitors?"
ShowToc: true
TocOpen: false
draft: false
---

## Offensive AI Con

The second week of October a dream of mine came true: the first [Offensive AI Con](https://www.offensiveaicon.com/). A few years ago when I was writing stuff like [this](https://hackbot.dad/writing/towards-autonomous-pentesting/), you would be lucky to find a blog post applying *any* ML technique to anything in the offsec space. This year, we had ~300 people in sunny San Diego who wanted nothing more than to meet up and compare notes.

Josh Saxe kicked things off with a characteristically [thoughtful keynote](https://docs.google.com/presentation/d/1D1gWFuT6AT3kLOqM1xl5YHKPvAhJh-VW/edit?slide=id.p1#slide=id.p1) that set the tone of the conference. It came from a very ML place: in order for a research field to agree on where it is in terms of capability levels, it's necessary to create the foundations. That means first-and-foremost difficult and agreed upon public benchmarks. After that: datasets and RL environments. For a mixed industry audience, Josh did an excellent job laying out the general roadmap of how a culture of capabilities develop. We know how these things go, and we can look at software engineering as a sort of 'older brother' domain shining a light on where AI security is. We can recreate their roadmaps and if we're diligent, their successes. 

The talks that followed ended up feeling in various ways like responses to Josh's keynote. Essentially everybody present, for the sake of empiricism, was measuring their agents/models/harnesses against _something_. To measure at all and leave the world of vibes is to be in the upper 5% of those experimenting with these tools at all. Off to a good start.

Fewer (but some!) of those talks were based on a _public_ benchmark. That is, one could write their own harness against the same set of tasks and in principle measure how effective their solution was in reference to the presenter's final score on that benchmark. 

Fewer still talks had a public harness - very few talks made it possible to run the same harness on the same tasks and reproduce the results, along with (for example) allowing one to try different models and compare their results holding the harness fixed. 

Further ahead and essentially absent was the creation of open datasets that can be used to train models to perform better on these benchmarks. I've [already written before on infosec's data paranoia problem](https://hackbot.dad/writing/infosecs-data-problem/) and it's likely outcomes on data science in the field. Presently I'm feeling pretty justified on that. 

So, a long way to go. As Josh said, we have to crawl, then walk, then run. Slowly and then all at once. Still, it felt _good_. Everyone at the conference seemed to understand what was being asked of them to move the field forward as practitioners and were eagerly plotting to see those asks fulfilled: go forth, make benchmarks, hillclimb them. Lots of back-slapping and big talk about what benchmarks we'd create and what environments we'd see completely saturated by the next time we met: the exact attitude you'd expect from peers looking at the green field work of the next few years and feeling _excited_. 

## The Fly in The Ointment

That pioneering can-do spirit sobered slightly by day two.  Let's get more specific. Who's going to make all of those benchmarks? And a better question, who is going to make those benchmarks at a company that will actually let them release them publicly? 

I was asked about this shortly after [my talk with Nick Landers](https://github.com/Offensive-AI-Con/OAIC-2025/blob/main/media/slides/day-two/nick-landers-shane-caldwell-benchmarks-to-breaches/slides.pdf). The question essentially went as follows:

>"If I put blood, sweat, and domain expertise into making benchmarks for infosec that are sufficiently challenging and easy to use, am I not just giving free capabilities to the labs and my competitors?"

I've got a lot of thoughts about that, and that's really what this post is about. But to sum it up: Yes. Yes, you are. But you've got to do it anyway or we might as well stop having cons.

First, I'll explain why I think the question asker was correct, and then we'll discuss why we have to do it anyway.

## The Dark Forest Problem of Evals

Evals, benchmarks, and datasets are not trivial to make. MMLU wasn't easy to make. ImageNet wasn't easy to make. It takes significant time, energy, and expertise.

To speak more to personal experience, [AIRTBench](https://arxiv.org/abs/2506.14682) was organically grown after roughly a year and a half of [Dreadnode](https://dreadnode.io/) making AI red teaming challenges. For each of these challenges, one or more members of the staff sat down to make something fun, difficult, and challenging for our users. Not all were appropriate for the benchmark: ultimately this resulted in 70 challenges. 

Someone wanting to sell an AIRT agent could take advantage of what from their perspective is free labor. They might turn this benchmark into a basic RL environment. Spin up GRPO, award 1 if the model is successful at a challenge and 0 if it fails, and let it rip. If motivated, they could set this up in about a week, and there would be little recourse to a) prove that it ever happened and by association b) get financial reward of any kind for the resulting product. 

Note the asymmetry: a year plus of careful creation of environments, versus a few weeks to plug-and-chug that into a model. By sharing the research publicly to encourage work in the space, the developer is announcing a benchmark to hill-climb on. The benchmark (designed as a test set, of course) ultimately becomes a training set. The second mover, the hypothetical person or organization that chose to wait until the benchmark existed expended no energy (and crucially, no currency) whatsoever until it was time to reap a financial reward. This second mover has the clear advantage. By doing the work and publicizing it, you've made developing capabilities cheaper. 

In academia, the deal is a bit more fair. As a researcher, evals can "make sense" in traditional incentive structures. With access to relatively low amounts of compute, benchmarks can be a good way to contribute to a research area you want to see investment in and get citations. If the benchmark becomes popular enough to end up on the model cards of the labs, you've got a good chance of ending up at one of the labs with the resources you want. Failing that, you'll certainly get a lot more people interested in working on research with you.

So _academics_ have some incentive to create evals. Do academics have the _capability_ to make the evals you want to see?

Cybersecurity is vast. We've got reversing, exploit development, EDR evasion, azure misconfigurations, malware development. That's just a small sample of the offense side. These are very particular skills that a relatively small amount of people know. Even getting the infrastructure together to run these kinds of challenges is involved and esoteric. Our field is dominated by practitioners. As we hill-climb on what we have, we will be forced to confront just how much there is to do and how few people there are to do it. It has to be us, because there's nobody else. 

Speaking for what I saw at Offensive AI Con: few public benchmarks showed up in talks. This is because folks were using models for whatever their day-to-day work task was that they knew best. They just weren't covered by the existing benchmarks. I don't think it's practical or desirable to wait for academics to save us.
## What if we don't?

Let's say we in industry don't make any benchmarks. What kind of world do we live in?

Well, every year we will get together for Offensive AI Con, Blackhat, Defcon, CAMLIS, whatever you like. We will greet each other warmly, and chat eagerly amongst ourselves about our latest crackpot schemes for offensive security agents. We will share what models we like best, and what has impressed or annoyed us lately. How about that GPT-6, huh? The worlds just not ready, one will say. They don't know like we know, another will respond. We will drink.

The next morning we will get up for talks. After a strong cup of coffee and a rousing keynote, we'll get into research presentations. Someone will present a novel use-case. They will explain their motivation for doing something as strange as what it is they've decided to do. They will show a bar chart. The x-axis will show many models. The y-axis will represent efficacy. You will nod along. You'll be a bit hazy on the details: you're not entirely sure what's being measured or how. That's okay, though. You know down is bad and up is good.

You will be pitched on some method or strategy. Maybe a training method, maybe a technique for dataset development, maybe a tool or harness improvement. You will get the gist.

As the talk comes to its climax, you will be shown a new slide. This will have a subtly different bar chart. You see, whatever the talk was about will appear now as its own bar. This bar chart will be higher than the bar charts you saw before. The difference may appear slight or vast. I can tell you without clairvoyance that it will be higher than the bar charts you saw previously. Since you know up is good, you know that this talk and the research it is based in has been justified and your time has not been wasted. The speaker stops talking, you clap. You'd love to interrogate this a bit, since that's what research is for, but it turns out the tasks are private and proprietary. That's okay, you've got the general idea. One or more of the ideas presented made the bar go up. 

The next speaker takes the stage. Yet another use-case you've never heard of. You're in unfamiliar territory. You're concerned you might be out of your depth. Not to worry though, as they move to the next slide, you find yourself looking at a comforting bar chart. This makes sense to you: down is bad and up is good. 

This isn't to say that the conference wouldn't be _valuable_. We were relatively low on public benchmarks at the first year of OAIC, and it was great. It's a big ask that takes a lot of work and places another potential barrier on smart people coming to speak freely about what they're working on. But, if we're serious about doing more than swapping war stories and enriching our own careers, we'll need to be serious about empiricism. In his keynote, Josh threw down a gauntlet. Dark forest be damned, we have to pick it up. 

## Evals || GTFO.

Fortunately, we've got a useful social construct from offsec's own culture to see us through. We are, by nature, a skeptical bunch in a field where social credit is deeply intertwined with _provability_. We don't value a theoretical exploit, or an exploit that runs on somebody else's machine, we value the producer of an artifact (code) that shows us how clever they are. It has to _run_. Then and only then we are happy to be enriched by the producers of that artifact telling us how it came to be, and we are satisfied that we can tweak it to our hearts content.

This, then, is ultimately a plea to organizers and reviewers of conferences that cover offensive AI: require a benchmark in submissions. If none exists for the use-case, eagerly accept talks that have constructed such a benchmark, however imperfectly, so long as it is _released_. 

To the leaders of the organizations that would have to approve such releases - it doesn't have to be a loss. In finding those who build upon and improve your benchmarks, you will find future hackers and researchers who care deeply about what it is you do. Consider it a public try out. If you can't strategically part with _all_ the effort and time associated with building the benchmark, find a subset you can part with and allow that to be published. 

And to the researchers: do it. Put out that north star, however imperfectly. No evals are perfect. The good ones are directionally correct. By working in this field as you have and experimenting with the technology, you have developed an intuition about what kinds of tasks in what kind of setup are most ripe for measuring the relative efficacy of models. Box it up, write it up, fight to release it. If it's not perfect, rely on others to make it better. But if it doesn't exist, there's nobody but you to work on v2. 

Ultimately, outside of competition with other startups and established security companies, we have a vested interest in pushing the field forward so that we can build useful tools and products that work for customers in the real world. You can hold tightly to the benchmarks you've built, but this will only stop your peers. If the labs want a benchmark badly enough, they will build it in house or find a private contractor to build it for them. Hiding your task-list from organizations with billions of dollars at their disposal will not prevent this. It is _you_, with limited computational and human resources, who would ultimately most benefit from a culture of public benchmarks.

I hope a year from now I'm writing about all the successes of the public benchmarks our community has made. I hope we're embarrassed by the collective riches we've handed over to each other and our peers in academia. I hope it becomes boring to talk about publishing benchmarks at all, because it's as normal as publishing PoCs. Instead of crawling alone, I hope we're running together.