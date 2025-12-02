---
title: "Twenty Billion Tokens of What, Exactly?"
date: 2025-12-01T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "training", "data"]
tags: ["llms", "training", "data"]
description: "Looking at the data and letting it look back at us."
summary: "Looking at the data and letting it look back at us."
ShowToc: true
TocOpen: false
draft: false
---

In my last post, I worked on getting a decent MFU for pretraining a 1B parameter model. In order to train it in a way that was practical for experimentation, I focused on the lower bound of Chinchilla optimality - 20 tokens per parameter. I chose my tokens by pulling a random subset of the 100B random subset of [fineweb edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-100BT). I wanted the data to be high quality, but I didn't spend any significant time thinking about what those twenty billion tokens represented. I wanted them to be non-random, so I could test the loss going down, and I wanted them to exist so I could go through an entire training cycle and get a sense of how many experiments I could run a day. 

But what's in those tokens?

The sheer size of the data involved in pretraining has been a constant discussion since LLMs got big[^1]. The quality of the data less so. If I were to capture the vibe of argument in the early 2020s, it would be: we want as much of it as possible from as many places as possible by any means necessary. Scrape Reddit, scrape GitHub, get ahold of as many books as humanly possible, and toss them all in the dataset. There are good tokens and bad tokens, but when you're scaling, they're all better than not having the tokens at all.

This was counter to what I'd known in deep learning to this point. I mostly worked in object detection and image search for a moving company and I spent the vast majority of my time thinking about data. SOTA performance was understood to essentially have nearly nothing at all to do with model architecture and everything to do with access to a high quality dataset, particularly one your competitors didn't have. 

What are my samples that have surprisingly high loss? Are the labels wrong, or do I just not have enough of them? If the model is performing poorly in videos that have extreme lighting, can I capture a synthetic data augmentation that makes the model performance invariant to those lighting conditions? Reports would come in from the business about errors in a weird case we cared about, and we made sure to capture those in the test set and tickets would get made about finding data or augmentations to fix this particular case. The vast majority of the day was looking at data, thinking about data, or looking for new sources of data.

We weren't alone in that, it was totally normal. The dream even for an only semi-resourced deep learning team was to get to Karpathy's level at Tesla. He was always upfront about spending a lot of dedicated time looking at the data. I had ["A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/) bookmarked, and I came back to the following passage regularly:

>The first step to training a neural net is to not touch any neural net code at all and instead begin by thoroughly inspecting your data. This step is critical. I like to spend copious amount of time (measured in units of hours) scanning through thousands of examples, understanding their distribution and looking for patterns. Luckily, your brain is pretty good at this. One time I discovered that the data contained duplicate examples. Another time I found corrupted images / labels. I look for data imbalances and biases. I will typically also pay attention to my own process for classifying the data, which hints at the kinds of architectures we’ll eventually explore. As an example - are very local features enough or do we need global context? How much variation is there and what form does it take? What variation is spurious and could be preprocessed out? Does spatial position matter or do we want to average pool it out? How much does detail matter and how far could we afford to downsample the images? How noisy are the labels?

I also found all that to be true. The first object detector my team trained for the startup was based on the [CoCo](https://cocodataset.org/#home) dataset. It had the most classes relevant to objects that would be in people's homes. It resulted in a pretty passable chair detector. It's also kind of terrible, like you'd expect data labeled by grad students to be, but for the most part it was functional. I remember a few weeks before launch I got feedback that we weren't doing very well on kitchen appliances. CoCo didn't have a "Kitchen Appliances" class, but we'd lumped all of the relevant ones from the dataset into a single class. I looked at a few mediocre results coming back from testers, and then went back to the dataset, and essentially found the distributions were totally different. Customers using the app were basically standing in the center of their room and doing 360 degree pan to capture their objects: the CoCo data for kitchen appliances had a large portion of them as the _subject_ of the images, captured lovingly from some sort of high angle shot.

There was nothing I could've done from the modeling perspective to fix that, and there wasn't a fancy data augmentation that would take us from the images we had to something representative of our inference distribution. I'm glad my career started with computer vision, because looking at the data was so _intuitive_. You could look at some samples and say something like "okay, if this is all I knew about object X, do I have a reasonable chance of identifying this?".

It makes sense to me that this data quality would still be really important, but you don't see a lot of people talking about it in LLM-land. Well, except for [Cody Blakeney](https://x.com/code_star/), who has been pretty upfront about it.

{{< x id="1940495038166741040" user="code_star" >}}

{{< x id="1774558322873495731" user="code_star" >}}


He works at [Datology](https://www.datologyai.com/), for which this blog post is an inadvertent advertisement. It was a combination of his tweets and Datology CEO Ari Morcos's [appearance on the Latent Space Podcast](https://www.youtube.com/watch?v=yXPPcBlcF8U) that got me interested in taking a look. 

Now that I'm interested in pretraining, it seems like I too should look at the data.

So, let's start with taking a look at [C4](https://huggingface.co/datasets/allenai/c4 ) and see what all those complaints are about. 

## C4: Common Crawl 

The first thing I noticed was how much advertising there is in this data. 


> `This 2013 Honda pilot EX-L is in excellent condition. Very well equipped with Leather, Sunroof, DVD system, Bluetooth, 3rd row seating, Alloy wheels, back-up camera, dual zone A/C and more..... Remote entry with multiple keys included. Special financing is always available here at Tropical Auto Sales..... Low payments and comfortable terms.... Come check us out! This pilot needs a new home! Price plus tax, tag, and $399.95 dealer fee.`


Just a really heinous amount of ellipsis. 

This was actually the most common thing I found with the C4 data. It's not surprising, you might imagine most of the internet as it exists. 


>`Unfortunately, Delle Donne had suffered an injury on her right thumb in a loss to the Washington Mystics. By the end of regular season, Delle Donne averaged **Be your own person**. It's a big reason why she is such a homebody who came home from UConn, because she craves to be around Lizzie and to experience Lizzie grabbing her and sniffing her and just spend quality time with her. Delle Donne scored 19 points in the victory. The previous record was held by Diana Taurasi and Seimone Augustus , who both completed the feat in games. With the WNBA's new playoff format in effect, the Sky were the 4 seed in the league with a bye to the second round. The Sky qualified for the playoffs for the first time in franchise history, earning the top seed in the Eastern Conference. Early life The daughter of a real estate developer and his wife, Delle Donne inherited her 6'5 1.`

Not really sure what happened there. Some error with the transcription.

>`http://player.vimeo.com/video/16500743Our”>http://vimeo.com/16500743″>Our Fearless Leader’s Opening Remarks at Crochet @ Cama 2010 from Karen”>http://vimeo.com/krwknitwear”>Karen Whooley on Vimeo.`

Mostly just web boilerplate, nothing I'd really care about the loss off. Maybe understanding "fearless leader" as a likely token pair.

At risk of spending a whole blog post cherrypicking examples of data, I decided to break these down into broad categories so we could look at the distribution of the dataset.

I decided on the following categories seven categories based on around an hour of clicking around the dataset. These are by nature extremely coarse, but should provide a broad understanding of the distribution of the dataset. 

**Educational**: Content that teaches something. Wikipedia-style text, technical documentation, stuff that reads like fragments of textbooks. The kind of thing you can imagine being "useful knowledge".

**Advertising**: Product listing, SEO content, marketing copy. Something might be "well-written" in that it's formatted well, but it's basically a product description. "Blogs" that exist primarily to get the reader excited about a product or service end up here. 

**Forum**: Anything intended to be conversational. Forum posts, comments, reddit style content. 

**News**: News articles of any kind. 

**Creative**: Fiction, personal blogs, jokes, that sort of thing. Recipes also went here, for want of anywhere else to put them[^2].

**Boilerplate**: General web boilerplate, fractions of websites, anything that reads like it's the written text thats been scraped from the header of a website. 

**Nonsensical**: Encoding errors, truncated text, text that might be one of the above but embedded in the middle of the document is web navigation or an ad. I also used this to include samples that were too short to express any sort of meaningful concept. 

I had Opus 4.5 vibecode this into a TUI for me, which you can find [here](https://github.com/SJCaldwell/hf-viewer) if you're interested. Looking through around 203 random samples, I ended up with the following distribution:

Advertising: 36%
News: 16.7%
Creative: 15.3%
Educational: 11.8%
Forum: 10.3%
Nonsensical: 7.4%
Web boilerplate: 2.5%

### Is More Always Better?

That's pretty heavily ad skewed. Not captured directly by the dataset is how arbitrary the samples felt. In fact, there was very little I would've _kept_ in the dataset. Many of the entries were fragments that did not in themselves contain a complete idea. Frequently I was looking at a sample that represented the beginning of a bibliography, referencing papers that were attached to no main idea. Looking at it on a "human" scale, there wasn't a tremendous amount of value here. The only documents that really represented _full_ ideas were the recipes.

The Chinchilla paper assumes that each sample is essentially the same. This is necessary for the argument of the paper, but is it true? It's difficult to believe that fragments of bibliographies and forum comments are providing as much value as a Wikipedia article. Certainly I wouldn't count them as _the same quality_ for any naive education context outside of LLMs. With LLMs, however, there's this sort of scaling maximalist argument. The average SEO content may not be educationally useful, but there's some learned compression about the style of these sort of documents that emerges that helps the final version of the model navigate the web or write marketing copy. And ultimately if the sample is truly not useful, and that compression isn't helping push the loss down anywhere among the 1 trillion parameters in the network, it will be "forgotten". 

This argument leaves out the realities of LLM training. If you're training on useless data, a high MFU becomes a lot less useful. Time to train goes up, reducing the number of experiments you can run for a given time, and the cost goes up too. Can we do better?

In the podcast Ari did with Latent Space, he brought up ["Beyond neural scaling laws: beating power law scaling via data pruning"](https://arxiv.org/abs/2206.14486), released in 2022, as research he considered foundational to starting Datology. 

The paper makes the argument that the some data points provide less information than others, and that if pruning is cheap, you can make better dataset decisions. The experimental results rely on image data, and propose a "prototypicality" metric. In an unsupervised manner, they perform k-means clustering on the embedding space of samples. Crucially, the number of clusters can be an order of magnitude off from the final models desired notion of classes without effecting the result.  Whether a data point is considered "easy" or "hard" depends on its cosine distance from one of the centroids of the cluster in embedding space.

This is intuitive. If a given sample is close to a centroid, it is probably common and lacks distinguishing features likely to trip up a model, and seeing that sample doesn't teach the model very much about the decision boundaries of classification. Samples of this type would keep training in "power scaling" range. Samples far from the centroid or "hard" are likely to represent difficult samples further out in the decision boundary, and the there's more to learn from the data point. This should push training dynamics closer to "exponential" scaling.

They find that this data pruning allows for cutting out a large amount of redundant data without impacting testing performance. That is, the models can train for less time and use less resources with the same downstream effectiveness on tasks we care about. 

Crucially, the point of the paper is _not_ the prototypicality metric itself: that's specific to the image classification task. If we take LLM pretraining as our goal, there's no simple map for producing centroids. So what makes the paper interesting for LLM training? 

The answer is largely in the theoretical framework. Let's back up. Imagine you have a pruning metric, and you can measure it's quality with a $\theta$ that characterizes how lossy it is. $\theta = 0$ would mean your pruning strategy was perfect, higher values indicates a lower quality pruning metric.

The paper has $\alpha_{tot}$ for $\dfrac{P}{N}$ where $P$ is the parameters of your model, and $N$ is the total number of training samples. The higher this ratio is, the more of a "data-abundant" regime you're in. We might think of this as having a fixed amount of information our model can learn, and we've got far more data then can fit in those parameters. 

$f$ is the fraction of examples kept after pruning, and $\alpha_{prune}$ is equal to $f \cdot \alpha_{tot}$.

They find that if you were to try to pick an $f$ without considering your $\alpha_{tot}$ (that is, decide on a fixed fraction of the data to keep without considering the dataset size with reference to your parameter count) you will end up with a training curve that starts exponential and then falls to power law scaling as the dataset grows. Meaning your pruning needs to be more aggressive as the dataset grows in size to keep exponential scaling. The more data you have, the pickier you need to get about the data you're selecting.

There's one caveat, related to $\theta$. $\theta$ is going to be an imperfect metric, but how imperfect it is defines your $f_{min}$. If your pruning metric is low quality, you will eventually start throwing out good data. At any nonzero $\theta$ as $\alpha_{tot}$ becomes large, you can't retain less than a minimum fraction of the data. Your test loss has a floor. 

Ultimately, the framework presents a hill to climb. Data pruning has a quality-dependent ceiling. A crude metric might get 2x data efficiency, a great one might give you 5x efficiency. Your rewards are bound by the informativeness of your pruning strategy. The goal is to find a pruning metric with low $\theta$ applicable to LLM pretraining that is cheap to calculate in a self-supervised regime. So if the prototypicality metric isn't used in practice, what is?

## FineWeb

To get a sense of the public state-of-the-art for data filtering, look no further than [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1). It opens thus:

>What is good data? This is probably the main question to keep in mind when creating a dataset. In most contexts and, in particular, in the context of large language model pretraining, "high quality" is not a very well defined term, and not even a property of documents that can always be clearly perceived through direct human observation alone.

The metric they come to is not quite the cheap, self-supervised pruning metric we would like. Instead, they ultimately rely on training small models and evaluating them on benchmark tasks that should start to be non-random relatively early in training. "Small" in this case is noted to be 1-2 billion parameters[^4]. The benchmarks they chose include CommonSenseQA, HellaSwag, MMLU, WinoGrande, and ARC. 

The described methodology for filtering the data is long, and I won't go into all the details (there's a perfectly good blog for that), but they used heuristic filters to cut down on common crawl, a hashing algorithm for deduplication, and _still_ found that the initial C4 dataset was better than what they had come up with. They then developed some statistical methods to develop heuristic filters, and ultimately created a dataset that was "better" as defined by an aggregate score over their evals.

How much better?

Well, a bit.

{{< figure src="fineweb_vs_c4.png" alt="" caption="Not as much as you'd think after all that effort." >}}

What's most remarkable here, to me, is how much engineering effort and talent went into the data to create something that only slightly beats C4. I don't doubt C4 is in itself a miracle, but _looking_ at that data and then seeing in practice that it's still one of the most well-cleaned ready made datasets for LLM pretraining is somewhat shocking.

Knowing a bit more about the effort that went into the dataset, I took a look at 200 samples by hand.

Advertising: 30.5%
Creative: 27.0%
News: 22.0%
Educational: 8.5%
Nonsensical: 5.0%
Web boilerplate: 4.0%
Forum: 3.0%

What isn't captured in this distribution is that the data looked better to me, in the 200 points I saw. There were less nonsensical fragments. More text that appeared "whole" in the sense that you could read it and it was coherent onto itself. The samples were also longer, perhaps due to one of the filtering methods the researchers came up with:

>Remove documents where the fraction of lines shorter than 30 characters ≥ 0.67 (3.73% of tokens removed)

I definitely feel like it was better, but from the samples I saw I'm not sure I'd want my 20B token budget to be allocated there, either. 

## FineWeb-Edu

FineWeb-Edu is a subset of FineWeb created by having `Llama3-70B` annotate 500k samples from fineweb on educational quality (rated from 1-5), and used that resulting labeled data to create a classification model. They then ran that classifier on all the documents, retaining anything that scored a 3 or above.

This seems almost hilariously simple compared to the rest of the blogpost. Not to say that it's not an impressive engineering effort and a cool model, but it's very classic self-supervised data filtering. Outside of the GPUs required, I would consider it very "easy" compared to all the other smart stuff they do in the blog. 

I took a look at the data to see how they did:

Educational: 74.6%
Advertising: 10.4%
News: 8.0%
Creative: 5.0%
Nonsensical: 2.0%

Pretty good! It's interesting to see where the model failed. For the ads, it seemed like some SEO content was so stylistically technical, or at least used enough technical language, to be considered educational. 

### Are Source Documents Optimal?

I liked most of what I saw in Fineweb Edu, but some things still bothered me. For example, there's a lot of artifacts of web scraping. 


>`|Skip Navigation Links|\n|Exit Print View|\n|man pages section 3: Networking Library Functions Oracle Solaris 11 Information Library|\n- produce an error message string\n#include <xti.h> const char *t_strerror(int errnum);\nThis routine is part of the XTI interfaces which evolved from the TLI interfaces. XTI represents the future evolution of these interfaces. However, TLI interfaces are supported for compatibility. When using a TLI routine that has the same name as an XTI routine, the tiuser.h header file must be used. Refer to the TLI COMPATIBILITY section for a description of differences between the two interfaces.\nThe t_strerror() function maps the error number in errnum that corresponds to an XTI error to a language-dependent error message string and returns a pointer to the string. The string pointed to will not be modified by the program, but may be overwritten by a subsequent call to the t_strerror function. The string is not terminated by a newline character. The language for error message strings written by t_strerror() is that of the current locale. If it is English, the error message string describing the value in t_errno may be derived from the comments following the t_errno codes defined in <xti.h>. If an error code is unknown, and the language is English, t_strerror() returns the string:\n\"<error>: error unknown\"\nwhere <error> is the error number supplied as input. In other languages, an equivalent text is provided.\nALL - apart from T_UNINIT.\nThe function t_strerror() returns a pointer to the generated message string.\nThe XTI and TLI interface definitions have common names but use different header files. This, and other semantic differences between the two interfaces are described in the subsections below.\nThe XTI interfaces use the header file, xti.h. TLI interfaces should not use this header. They should use the header:\nSee attributes(5) for descriptions of the following attributes:`

On the one hand, what are you gonna do? You're scraping the web. There's bound to be web stuff in there, and a lot of the navigational/header type stuff is just there. On the other hand, do we really believe the weird formatting and web artifacts aren't impacting the educational quality of the samples?

The average sample quality was _much_ higher, but there are a lot of samples that still seem incomplete, and look like it would be fairly easy to rewrite them to be more explanatory, or cleaner. Obviously it's entirely impractical to do that for a multi-terabyte dataset, even with a legion of grad students. 

Oh, unless you had a robot do it, I guess. That might work.

## Send in the SYNTH

The narrative around synthetic data has also changed fairly dramatically in the last year or so. There was this narrative going around that a model consuming outputs of another model as part of its training was essentially poison. The thought went - any data pulled from a crawl post the release of ChatGPT might well be a model. It would be extremely difficult to distinguish this data, and the data was likely to have very low value. The hallucinations and general schlubby style of the output would be compounded in future training runs, and the models would inevitably get worse. 

That's turned out to not be the case. In fact, synthetic data has become something of an expected cornerstone in model training that modern data teams are required to be familiar with. 

[Phi-3](https://arxiv.org/abs/2404.14219) from Microsoft used synthetic data:

>In our previous works on the phi models it was shown that a combination of LLM-based filtering of publicly available web data, and LLM-created synthetic data, enable performance in smaller language models that were typically seen only in much larger models.

[Kimi-K2](https://arxiv.org/abs/2507.20534) used it:

>A key advancement in the pre-training data of Kimi K2 over Kimi K1.5 is the introduction of a synthetic data generation strategy to increase token utility. Specifically, a carefully designed rephrasing pipeline is employed to amplify the volume of high-quality tokens without inducing significant overfitting

[Olmo3](https://www.datocms-assets.com/64837/1763662397-1763646865-olmo_3_technical_report-1.pdf) used it[^5]:

>We introduce Dolci Think SFT (§4.2), Dolci Think DPO (§4.3), and Dolci Think RL (§4.4), new cutting-edge post-training datasets designed to target a broad range of key capabilities such as math, coding, instruction following, and general conversation. The dataset includes synthetic examples with long thinking traces for supervised fine-tuning, high-quality contrastive data following the insights from Delta Learning Geng et al. (2025)...

So, synthetic data, very hot right now, etc. But there's one _pretraining_ dataset in-particular that I'm most excited about: [SYNTH](https://pleias.fr/blog/blogsynth-the-new-data-frontier).

If you've been following [Alexander Doria](https://x.com/Dorialexander) on Twitter, which you should be, you'll have seen a lot of synthetic data posts. He's the person I've followed who's most focused on taking [Physics of Language Models](https://physics.allen-zhu.com/) seriously. One outcome from that school of thought is basically that you should be able to _engineer_ data that will allow a model to efficiently learn the skills it needs to succeed in the inference environment you're interested in. You can also connect these ideas to the "Cognitive Core" [Karpathy proposed](https://www.youtube.com/watch?v=lXUZvyajciY): essentially finding the minimum number of parameters required to bake in basic world knowledge and the skills to search for what is needed at test-time "live". SYNTH is the result of pushing those ideas forward. 

SYNTH starts with 50,000 Wikipedia articles and expands them into problem and resolution paths. Essentially turning those articles into the context for different skills being tested. To cut down on the inference costs of generating this data, relatively small open models (Qwen 8B, for example) are used to generate the data, with a complex harness around it to verify reasoning traces and improve the quality of the data. 

The pipeline is fairly complex, and Pleias has promised to give us more information on it in the coming months, but to get an idea:

{{< figure src="synth_playground.png" alt="" caption="Lot going on here." >}}


That's a lot of hubbub, what does the data look like?

The data looks really, really cool. It's basically designed for user interaction from the jump. Each sample has a query, that was generated from one of the documents from an LLM. So for example, for a query based on [Kolmogorov equations](https://en.wikipedia.org/wiki/Kolmogorov_equations), for a task `memorization`, the query might be `so like, how does the thing with states moving between each other work over time, i mean do they always change or can they stay still?`. Then there's a generated reasoning trace, which looks pretty trippy. I'll show a partial one, because they're pretty beefy:


{{< figure src="synthetic_traces.png" alt="" caption="" >}}

Note the special symbols, which have all been added as meaningful reasoning traces that the model is trained to use. 

And at the end, you get an answer. 


> `In Markov chains, states can indeed stay still for periods of time, but they don't necessarily remain unchanged indefinitely. The mechanism works through what's called jump processes - there's an overwhelming probability that a state will remain unchanged in a small time interval, which means states can persist without transitioning. However, when changes do occur, they can be quite significant or radical. Think of it this way: states tend to stay put unless something triggers a transition. When transitions do happen, they can jump the system from one state to another in a more dramatic way than a gradual change. This creates a system where states spend most of their time in one condition, but periodically experience more substantial shifts. The mathematical framework that governs this behavior involves complex equations that describe how probabilities evolve over time, but the basic principle is that persistence and change both play roles in how states move through the system.`                    

It looks entirely like an instruct style dataset, but with deep reasoning traces built in. It's intended to be used right off the bat with pretraining data. I won't bore you with a breakdown of the distribution: there are no ads in this dataset. There's mostly different tasks like this, along with some samples meant to help the future models explain details about itself (which I'd file under creative, I guess). 

The resulting models trained on this seem to perform quite well, reaching a non-random MMLU score at 10B tokens. 

{{< figure src="synth_training.png" alt="" caption="Insanely quick non-random MMLU" >}}

It's easier to imagine successful pruning metrics from these synthetic playgrounds. You can essentially generate as much data as you want - and you have a lot of information about where it came from, and a rich possibility of verifiers for filtering. If you were to extend this to LLMs working with tool-use with data filtered using RL-style verifiers, there's a lot you could do.

I don't know that synthetic data will go on to dominate large pretraining runs, but I certainly believe best in class small models will invest heavily in synthetic data.

[^1]: Both in terms of literal number of parameters and the hold on the cultural and scientific consciousness of planet Earth.

[^2]: This was a weirdly large part of the samples I looked at. Like, probably the most common content that wasn't an ad.

[^3]: This is still only a fraction of a percent of the full `en` dataset, but 10,000 felt like a better representative subset.

[^4]: I'm not jealous, you're jealous.

[^5]: A lot, actually. The word "synthetic" shows up quite a bit in the technical report. I only included an early example for brevity.
