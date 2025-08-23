---
title: "RL Needed LLMs Because Agency Requires Priors"
date: 2025-08-19T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "rl"]
tags: ["llms", "rl", "research"]
description: "We tried RL once. It didn't work. I'm confident it will this time."
summary: "We tried RL once. It didn't work. I'm confident it will this time."
ShowToc: true
TocOpen: false
draft: false
---

I'll begin at the end: RL works better for LLMs than it ever did tabula-rasa or behavioral cloning. The world is, by default, partially observed. Information is imperfect. Strong priors enable you to overcome this. It's difficult to get stronger, more transferable priors than just grabbing a multi-billion parameter compression of all the data you can find. 

That's a strong statement. I'm going to justify it via a rambling retrospective, so I thought I ought to motivate you to read through it. 

As an AI researcher, I'm focused primarily on LLMs. This is probably no surprise to you - 99% of people doing research are currently LLM people. That's where the money is, that's what's getting turned into products, and that's where the energy of open-source is. To focus on anything else you need to be particularly motivated.

Working in this field as a researcher or engineer, it's common when talking to others in the field to talk about _when_ you realized LLMs were going to be a big deal. We're all on the shoulders of giants but if you can say, for example "Oh, I was into attention before the release of GPT-2" that's better than "GPT-4's release was a big moment for me"[^1]. My story does not sound great. I was following the AI research broadly, but had a particular distaste for LLMs. That being the case, it took me a little while to catch up.

Here, I want to provide some historical context and talk about why I wasn't motivated by LLMs, what's changed, and why I was wrong. 

I was a web-app pentester for about two years in 2016-2018, fresh out of undergrad. I had a background in bioinformatics that I was choosing not to use in order to stay as far away from academia as possible[^2]. I'd taken a few classes in security and read [Hacking: The Art of Exploitation](https://en.wikipedia.org/wiki/Hacking:_The_Art_of_Exploitation) along with [The Web Application Hacker's Handbook](https://www.oreilly.com/library/view/the-web-application/9781118026472/). The first six months or so were mostly a continuing undergrad with a provided salary, complete with poor work-life balance and a lot of studying, but I was pretty enamored with the work and confident I'd be happy doing it for the rest of my life. 

That said, I was 22, and it turns out life is long. At the end of those six months I realized how much I had to learn, but also how much slower my learning process was. In a five day web app test, my schedule could be broken up as follows.

1. Monday: Explore the application and map out all of its functionality. Build up the auth matrix I'll be testing permissions against later. Set up an authenticated Burp Suite scan and get it kicked off.

2. Tuesday: The client provided a staging deployment with two web workers and half a can of Red Bull, so there's very little to do but scale back the threads and triage results coming in. Hopefully that finishes today.

3. Wednesday: Scans done. Go through the rest of the manual checklist, mostly authentication/authorization type checks. Wrap back around to any responses from the scan that weren't directly exploitable but seemed weird enough that you won't let it go without getting your hands on it.

4. Thursday: Here's the great day. You've checked for everything you _have_ to check for and feel confident about it. Now you're off the checklist, and you're sniffing out all the weird parts of the application. Every app has some functionality that feels way less "stock" than everything about it and custom development means custom bugs. When I did something I was proud of, it was Thursday.

5. Friday: Show's over, it's time to report. Make sure you've validated and collected evidence for everything before you lose access to the environment, show mastery over the English language and make the reproduction steps you're not confident anyone is ever going to read much less follow crystal clear. Ship it.

After that initial learning phase, I realized I basically just really enjoyed Thursday. That's where I felt like a _real hacker_ and not like a guy executing a checklist. Thursday is one day and there are four other ones, so this wasn't very satisfying. I wanted all my days to be Thursdays. I became interested in automation, and looked around at all the tools that existed for it. There were many clever tools for every element of web app testing, mostly taking advantage of the regularity of HTTP messages and their contents and doing things with regular expressions I honest-to-god didn't know you could do. But having a machine learning background, they seemed brittle and limited. A list of 1000 regular expressions split among 20 plugins is great and all, but what about some classifiers? I started studying for my OSCP around this time as well, and the heavy recon focus did nothing to disabuse me of the notion ML should be involved[^3]. 

This was my frame of mind and the kind of problems I was thinking about when I heard about AlphaGo. The competition was over at that point and I could freely access [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961). It blew my mind. I was familiar with supervised learning, but maybe had one lecture on reinforcement learning and hadn't understood what it could be useful for. Now they were combined in this beautiful way and I saw a light at the end of my tunnel. Go is a very difficult game, following a checklist to test webapps is less so. If it could do one, it should be able to do the other. The thing that was most attractive about RL (and still is) is the direct optimization for performance on the task I cared about. Why should I have all my human code taking action based off of classifiers when I could just have it do the thing?

I wrapped up my OSCP, turned in my two weeks notice, and went back to grad school, sure I would find a way to use deep RL for penetration testing. My confidence increased further when [AlphaZero](https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) released. There you go! The recipe generalized outside of Go. It generalized to all sorts of games.

I wasn't alone in my excitement; far from it. OpenAI was [all over it](https://spinningup.openai.com/en/latest/) and applying it to everything from [robots solving rubik's cubes](https://openai.com/index/solving-rubiks-cube/) to [Dota 2](https://openai.com/index/openai-five/). There were pesky technical details dogging everyone, but the general vibe was "they'll be ironed out if we throw more research at it". Here are a few technical details we were carefully ignoring, some of which seemed like they could be overcome and some less so. The recipe seemed to generalize well to _perfect information games_. So what's missing? You may notice some themes:

1. Most environments you would like an agent to operate in _do not_ have perfect information. Pentesting obviously does not, 90% of the game is reconnaissance. You've got what is functionally a black box, a picture of which you sketch out by "asking questions" in the form of scans, requests, and various other tools that involve actively gathering information from a target or passively finding it elsewhere. Even when you get as complete a picture as possible externally, you're still far from perfect information. Progress was made with [Poker](https://arxiv.org/abs/2007.13544), a game with imperfect information (but crucially imperfect information you know)

2. Action space design is painful and full of human priors. Board games are one thing. Encoding tic-tac-toe moves as vectors is pretty straightforward. Chess requires some creativity, but you can get there. Then you get to Dota 2. At that point you experience [pain](https://arxiv.org/abs/1912.06680)[^4]. 

3. Environment representation is painful and full of human priors. Beyond the fact that you have to figure out how to represent everything as vectors, what's really necessary? How are you going to present a web app state as a fixed-size matrix?

4. Designing reward functions is really hard and full of human priors. In particular if you're doing tabula rasa RL. Any impressive agentic thing you can imagine is just not going to happen from taking random actions. So partial reward functions were used to award the model for going in the vague direction of right. Reward hacking is bad now, reward hacking was so much worse. The most infamous, visually engaging example is probably [CoastRunners](https://www.youtube.com/watch?v=tlOIHko8ySg). It's a racing game. OpenAI provided partial reward for the agent picking up a powerup that gave you a speed boost. This seems super reasonable, since going fast is likely to get you to win, right? In this case, the agent finds a loop where it can just pickup speed boosts and wirehead itself without ever doing the thing you wanted to do. People were so worried about reward specification problems! It was a non-negligible part of why the AI safety people were going nuts.

5. Collecting data is hard. You'd like some supervised data to get some good priors, but your environment and action space are some kind of unholy abomination that only works in the weird framework you made up, so you have to synthetically generate it yourself if you get it at all[^5]

Then there was the specter of Yann LeCun, taunting us. 

{{< figure src="yann_cake_taunts.jpeg" alt="A slide from a Yann LeCun presentation in NeurIPS 2016" caption="It still hurts" >}}

Most of those up there come down to pushing humans _into_ a loop you would like them out of. I don't mean the way we talk about now, like "Claude Code has a human in the loop because I have to ask him to fix stuff", I mean the deep learning process itself. Neural network architectures represent in some sense, the priors in place on the search space they're free to optimize over. Ideally you want it to be convenient to find good solutions, and a lot of deep learning tricks back in the day were just that. How do I set my tabula rasa parameters so they're likely to end up in the good place? How do I make sure my gradient steps are big enough to get out of bad local minima but not so large I never find a good local minimum, etc. 

RL has this whole other part to it, where you're defining these very key parameters that are deeply encoded into what the network can consider and do. If you don't provide an action for it, the agent can't take the action for it. If it's not wrapped up in the environment representation, the network is blind to it. You now have the priors you set running headlong into engineering realities and compromises. It's hard, and you're _very_ involved and iterating on it is _very_ slow. Ultimately, you'd want this representation to be something discovered by the deep learning algorithm. We sweep the hyperparameters for 2% gains, for god's sake, why would I want a human being to be involved in the most fundamental basic representations of the problem? That's what we learned from computer vision and natural language - provide the rawest representation possible of the data and let the model figure out what to do with it.

All of this seems obvious in retrospect because we just have better options now. At the time, it seemed like the best game in town and like something would just fall into place. Maybe instead of hand-designed environments you'd just have _x_tovec for whatever your environment was and you'd learn a dense representation of it in an unsupervised way and that would be fine. Maybe instead of a reward function you'd use a reward model trained on [human preferences](https://arxiv.org/abs/1706.03741). It seemed feasible! 

## My Experience with Deep RL: Metasploit Gym

Post graduation I started work as an ML eng mostly doing object detection and image search. This was working with neural nets in prod, which was great, but had nothing to do with agents. When the pandemic happened I found myself with a lot more free time on my hands, and I used a lot of it to read the existing ML offsec literature. There wasn't a lot I was crazy about. Those systems that did use RL appeared to be largely simulation driven. Simulation is a big word that can mean a lot of different things - I'm not _anti_-simulation, but a simulation is only as good as its fidelity. Most papers would set up a graph of nodes that represented "attacker boxes" and "defender boxes". Then they'd have different "attacks" that had particular percentages of success. We're talking really high level stuff, like one action might be an "SSH exploit" action that had some percentage chance to succeed if the defender box had an "SSH attribute". 

My issue with this is very simple - if you can't take that trained model and swap the actuator for your sim to something that takes action in the real world, I'm not interested. You're just setting up a system to see if an RL agent can learn ideal strategies for your hermetic world model. I sure hope so! That's what it's for.

So while I was being mad and reading simulation papers[^6] I came across this paper: [Autonomous Penetration Testing using Reinforcement Learning](https://arxiv.org/abs/1905.05965) from [Jonathon Schwartz](https://jjschwartz.github.io/)[^7]. I flipped through it and found it was all in simulation, and was preparing myself to get mad again. This section struck me, though:


{{< figure src="abstraction.png" alt="An excerpt from page 16 of the above mentioned thesis" caption="Pentesters do in fact be using high-level tools." >}}

The simulations people were making were in fact pretty simple and "high-level" but were necessary to make the problem tractable with RL. However, hacking tools were already in a sense about making things high-level in order to make it easier to do your job. From this, I basically ignored the simulation part and locked in on the "metasploit is a high level API for hacking" thing, and designed [Metasploit Gym](https://github.com/phreakAI/metasploit-gym) around that. I gave a talk on that [here](https://www.youtube.com/watch?v=EiI69BdWKPs) if you're interested, and a [blog](https://hackbot.dad/writing/towards-autonomous-pentesting/) that goes in depth on what I was thinking at the time. Mostly though I want to use this space to talk about all the problems I ran into.

### Action Space
Just a total bear. I had this idea that every metasploit module would start with all the defaults, and could be applied to a particular "service". This worked for basic stuff, but was a huge flattening of the actual potential action space. Running an nmap scan, for example, involved picking defaults for all the scan parameters and hardcoding them. That allowed it to work, but now there's loads of behavior that my agent couldn't express. A lot of the power of frameworks like Metasploit is how configurable the modules are. It couldn't be more or less stealthy, it couldn't look for specific things on specific boxes, it was just "scan". That same basic problem plagues most of the actions. 

### Environment Representation
I essentially chose to represent boxes as vectors and networks as a matrix. So every service could be one-hot encoded for whether it's on or not. You've got HTTP open? That's a 1 for the HTTP service section, and so on. I didn't have a way to represent multiple services of the same type, nor did I have a way to surface the version information a scan would provide. I had a vague idea that I could replace the one-hot encoding with a 0 if the service wasn't on, and a dense word2vec representation to provide more information, but that's still pretty limited.

The network matrix itself was also of fixed-size, meaning there was a maximum number of hosts I could be aware of. If there were less hosts than columns, no big deal, those columns would all stay zero. If there were more? Uhhh. Train a different model, I guess. "Future work". 

### Reward Function Design

This was potentially the most painful part. In a perfect world, you design a reward function wherein the model is rewarded at some scalar for having done the thing you wanted it to do. In the simplest case with Metasploit Gym, root the box and get some data off of it. Unfortunately, if you don't get _any_ reward signal, you can't learn. Randomly choosing actions in our action space means running random modules on random services of random hosts. The vast majority of the time, nothing at all happens.

So you need to provide partial reward for something that feels _in the direction_ of the thing you actually would like to provide reward for. Dumb stuff that happened to me:

1. I provided reward for scans. Scans are information gathering, and we like recon. Immediately I got reward hacked because each scan got the same amount of reward, and you could wirehead by just scanning all the time. Updated this to only provide reward if new information came in.

2. Ditto on exploits. Initially had a function for rewarding a successfully run exploit based on the privilege level you got from the resulting shell. Wireheading again, fully rewrote the environment update and reward logic to look at the diff between the previous cumulative environment/privilege level state and the new one in order to determine whether any reward was due. This got ugly.

3. Initially the scan was async. The action kicked the scan off and when it was done the agent got that information at whatever timestep the new environment information was available. I didn't have logic to go back and assign the reward to the action that had actually done the kicking off, and so the reward just got glommed onto some totally random action. Agent immediately zeroes in on that action, despite it having nothing to do with the reward. Quickest fix was making the scan synchronous, which was slow.

Which is to say, everything they tell you when you read about RL happened. It was honestly really fun to work on, but I couldn't help but feel how much of _me_ was being wrapped up into the representation of the environment and the calculation of reward. That doesn't happen when you write an object detector. All my abstractions were sitting between what I wanted the model optimized to do and how its world was represented. 

### A lightbulb that took years to go off

StrangeLoop, where I was to present the results of the MetasploitGym was fast approaching, and the model was taking too long to train. There was a lot of basic stuff it was struggling with. Particularly frustrating was how many pointless commands it ran - exploits designed for specific WordPress plugins being shot against SSH services, that kind of thing. Just stuff a person would never do. It made sense given I was starting from nothing, but it wasn't helpful.

It occurred to me that what I wanted was stronger priors. Generating supervised data was going to be hard - even if I wrote code to process my history running metasploit into a supervised format, it would just take too long to generate as a person for my timeline (about a week). So I implemented a simulation, essentially borrowing everything I'd read in all those papers that had made me mad. If an action had no chance of being successful, it got no reward ever. If it was roughly the correct service, it would get reward some percentage of the time. It was more likely to get a reward if it had already scanned and was aware that service was really up, that kind of thing. This allowed me to run purely in simulation for awhile and get those reasonable priors baked in, and I could do the rest of the training in a real environment with the same action/env space with just a different actuator. This allowed me to get a reasonable model for a demo done in time for my presentation[^8]. 

At the end of the day the results were fine. Random policy solved the box in 20% of 100 episodes tested. Simulated pre-train was roughly 40%, and the future episodes I ran for training weren't a waste of time. When the policy converged it was capable of rooting the box in 100% of the episodes. I was happy and thought I'd done something clever, regardless of the laundry list of limitations I described above (and training on test).

## LLMs

In the meantime GPT-2 is happening, GPT-3 is happening, and I am mostly not interested. People trying to build products out of few-shot prompt autocompletions bugged me. I didn't like the loss function! Probably because it smelled too much like Yann's dunking. Also, I was just mad that people at OpenAI were working on this instead of trying to fix all the above-mentioned problems of RL. 

When GPT began to be instruction tuned, I could see the value a little more. Working over text still seemed very awkward to me[^9], but more promising.

Once the UI for ChatGPT was released, I decided to give it a swing on solving Hack the Box challenges. Similar to a lot of experiments I did at the time, I just asked it to respond with what I should paste in the console next, and I returned the reply to it, and I just ran it until it fell apart.

{{< figure src="htb_feb12_2023.png" alt="One of my first ChatGPT interactions, wherein an nmap command comes out" caption="That command did not, in fact, need a predetermined action space" >}}

It was a remarkable experience. Commands to install various recon tools just came dropping out. It would see a web service, install gobuster, run it, and start poking around various parts of the application. It didn't solve any Hack The Box challenges, but the recon was reasonably solid. And look what all those _priors_ could do! 

Even in this infantile state, it was just super clear that nothing I'd done in Metasploit Gym could even compare. There was just no way with the action and environment space I'd written could come up with the commands I was getting out of a model that had been in no way trained to perform the penetration testing task. Maybe if I'd made the action space like, at the character level in a terminal? But that was just tokens but worse. 

If you wanted more supervised data, that also seemed really reasonable. In fact, it would be basically a cleaner version of what this model was trained on. You would want to collect terminal logs, maybe annotate them a bit. I was very sad about RL not being part of it, but it was just so damned flexible. Even before it was multi-modal you had [natbot](https://github.com/nat/natbot) making these simple textual representations of the browser contents to interact with web applications. I [forked it](https://github.com/SJCaldwell/phreakbot) and `text-davinci-002` was ripping through the Damn Vulnerable Web App.  

Philosophically, I was still annoyed. It was nice to know RL was still useful in RLHF, but that wasn't really what I wanted. There was nowhere in the stack that models were being tuned directly from the objectives _I_ intended for the LLM to be good at. The open model ecosystem improved a lot, and I could freely SFT open models for tasks I cared about. The loss function was still token prediction, though. I couldn't directly optimize for, say, "hacking". There was a lot of research community disagreement over whether that mattered. As the models got larger, they seemed to just _get better_ at just about everything. Bar charts were going up. Next-token prediction on transformers was an extremely scalable paradigm and the research investment yielded a huge amount of positive results, so why go back to small scalar rewards and the painful training dynamics of RL? 

Philosophically annoyed or not, it's hard to argue with the evals.

### RL Comes Back

The models improved at a steady rate throughout 2022-2024. Harnesses that used to need to be totally complex to get a particular behavior could be relaxed. Tool-calling was now very normal, and you could expect structured outputs without a lot of fuss. I still believed that genuinely novel search‑and‑exploit behavior would be limited without real RL. There's a lot you can do as a lossy compression of high quality SFT trajectories, but were we going to see models perform novel behaviors of interest[^10]?

I'm not a lab insider, and can't comment on when RL outside of RLHF became a priority for the labs. Like most people, I was introduced to GRPO (Group Relative Policy Optimization) with the release of [DeepSeek-R1](https://arxiv.org/abs/2501.12948). I was, however, incredibly stoked. The models now had a reasonably easy to implement algorithm that let them touch grass with base reality and see real improvements from it. 

Shortly thereafter, [Will Brown](https://x.com/willccbb?lang=en) released his infamous [gist](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb). I've never seen a gist with 1,288 stars and 386 forks before. I've also never seen a gist with a BibTeX citation in the top of the docstring. If a gist ever deserved that, though, it was this gist. It made the research extremely accessible to a ton of people really quickly. It's simply infrequent that you can experiment with the results of a tome-like research paper within a few weeks of its release on a free Google Colab.

The task is [gsm8k](https://huggingface.co/datasets/openai/gsm8k). Let's take a look at a few of the reward functions:

```python
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
```

Pretty reasonable. You want to assign reward if you get the correct mathematical answer.

So what about the rest of them?

```python
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]
```

This probably made me happier than I had any right to be. Here was a _partial reward function_ in 2025. Instead of generating hundred or thousands of the appropriate trajectories, just nudge the model with a reward function. In principle there was no reason why this reward had to be calculated from the data instead of from some external reward.

Everything we used for evals might be directly applicable. Passing unit tests, CTF flags, whatever stable signal you had in your environment from a task was now fair game. It took off immediately. I think because the basics were super easy to grok (thanks Will) and because it felt like it put people who had domain expertise to write good reward functions and construct good environments in the driver's seat of the tasks they care about without the gargantuan task of dataset collection and cleaning.

RL does introduce some painful infrastructure problems. Scaling up environments isn't easy, but it's in clearly doable in principle. Labs like Nous have spun up frameworks for [asynchronous RL](https://github.com/NousResearch/atropos) with plug and play environments. [ART](https://github.com/OpenPipe/ART) is doing an incredible job making the training itself very easy. The recipe hasn't been canonized, yet, but it will be in a few years. That's not to say it's not extremely difficult, just that it's now doable. You can grab one of the increasingly capable open models off the shelf, and if you put in the elbow grease to create difficult, realistic environments, you can train a model directly on the objectives you care about using RL. It's very exciting. Everything old is new again, and there are tons of papers to be written where you take something that worked for Deep Q-Networks (DQNs) and figure out if you can make it practical or useful for LLMs. We all get to talk about credit assignment again.

## Conclusions: Agency Requires Priors

The book isn't closed on RLVR (reinforcement learning from verifiable rewards). Nathan Lambert from AI2 said on the [Latent Space](https://www.youtube.com/watch?v=PAz_-xPJcRM) podcast a few weeks ago that he wasn't including a ton on RLVR in his upcoming RLHF book because it'll be years before the research solidifies enough for a book to be written. Without speaking to where it might go, I just want to talk a little bit about how different training LLMs in these paradigms _feels_ compared to that [Metasploit Gym](https://github.com/phreakAI/metasploit-gym) work.

The action space and environment space have just opened up to an insane degree. Tools (actions) can be modified without any change in the underlying code running the model. This is also true for the environment. You can represent whatever you want through text and images in as raw a form as you like. The limitations are around what you can bring from the environment. The demo environment you set up can grow to be more mature, there's a ton less for you to think about. This experience of trying to map to matrices just isn't a thing. I think that explains a lot of the agent demos you see on Twitter - it's just ludicrously easy to write up an API for a tool nobody has given a model access to before, run it, see something cool, and post it.

The priors are also just stupidly powerful. If your model is trained to use tools, it will use your tools. If your tools enable a task to be solved, it's entirely plausible you don't even need to write a partial reward function. The reward hacking that falls out of trying to coax a successful episode out of a tabula-rasa model is just not a thing you have to engage in as often. If you can evaluate it, you can reward it. Many evals - unit tests, CTF flags, compile/run checks, reconciliation diffs - are already verifiable signals. LLMs + tools surface the state; RLVR converts those checks into training signals. If you want to hear more about the benefits of evals, (and why you should write your own) I speak on that [here](https://hackbot.dad/writing/agony-and-ecstasy-evals/).

That's how I think about LLMs now. This giant collection of priors and inductive bias that provide a really beautiful general starting point for whatever task you want to do post-training on. It's on us to figure out how to design and deploy the environments this reward signal will come from in a scalable way, but it feels like a little elbow grease in comparison to the myriad of things holding us back in 2019. 

So, maybe Yann was right after all about RL. We just didn't predict we'd be given a cake covered in frosting and given the enviable task of figuring out how to put the cherry on top.

[^1]: Crucially, note that this _does not matter_ and mostly has nothing to say about somebody's intelligence or research intuition. This is purely a social game we play amongst ourselves. In another life we would be comparing front lawn products, or something. I'm not saying I don't participate, I'm just saying it's a dumb thing to do. 

[^2]: Loved the analysis, but Biology is so violently slow and frequently irreproducible that I think it would've killed me. Popping shells provides the more immediate feedback I need to function. 

[^3]: Yeah, man, you gotta run like 1000 scans and then read them over and over again until you develop an intuition for what's worth triaging. That's classification! You're making me a classifier!

[^4]: To be clear, I deeply admire this work. This paper was my coping mechanism whenever I couldn't think of a way forward on pentesting. The fact that there were compromises involved in the action and environment representation are just showing how killer engineers made the research they had stretch to the agent they wanted to make. It's awesome.

[^5]: This is a clue that will help you later!

[^6]: Most of my research ideas come from this. That's probably true for a lot of people.

[^7]: He's not really into the infosec domain anymore, but I still like to shout him out. He answered my emails back in the day and just seems like a bright guy. Thanks Jonathon!

[^8]: That, and my buddy Grady's home Proxmox lab. Thanks, Grady! Thanks, Proxmox! Truly never seen faster environment resets in my life. I literally didn't even implement logic to check whether the reset was done before the next episode started because Grady's Proxmox server was so fast. 

[^9]: Honestly even funny to remember that was a problem seeing how good structured output has become. 

[^10]: There's an argument made that you wouldn't _need_ that in order to be economically valuable. Obviously it was true to an extent, because a huge amount of people invested in the advantages of "semantics-aware programming" that the models provided in order to make startups across all sorts of interesting verticals. I don't want to see economically interesting behavior operating at the average of a human-generated training set, though. I want to see AI generate novel exploits.

[^11]: I had missed [DeepSeek Math](https://arxiv.org/abs/2402.03300) entirely. 