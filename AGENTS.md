# AGENTS.md

This file describes how to maintain Shane Caldwell's Hugo research blog.

## Project Overview

The site is a Hugo blog focused on AI security, autonomous agents, evaluation, machine learning systems, and offensive security. It uses the vendored PaperMod theme with repository-level templates and styling.

Important paths:

- `hugo.yaml` — site settings, navigation, homepage research highlight, and Hugo configuration.
- `content/writing/` — long-form posts.
- `content/papers/_index.md` — public research and publications.
- `content/talks/_index.md` — conference talks.
- `layouts/index.html` — custom homepage.
- `layouts/writing/list.html` — year- and topic-organized writing archive.
- `layouts/shortcodes/paper.html` — research-card markup.
- `layouts/partials/extend_footer.html` — article TOC, footnote, reading-progress, and code-copy behavior.
- `assets/css/extended/research-blog.css` — the complete custom visual system.
- `static/CNAME` — preserves the `hackbot.dad` custom domain in generated output.
- `public/` — generated Hugo output. The GitHub Actions workflow rebuilds this directory for deployment.

Do not edit files under `themes/PaperMod/` for site-specific changes. Override the theme from the repository-level `layouts/` and `assets/` directories instead.

## Development and Verification

Run the local preview:

```bash
hugo server --bind 127.0.0.1 --port 1313 --disableFastRender
```

The site will be available at `http://localhost:1313/` with live reload.

Generate the production site:

```bash
hugo --cleanDestinationDir --minify
```

Before committing, also run:

```bash
git diff --check
```

Responsive changes should be checked at a phone-sized viewport around 390px wide and at a desktop viewport. At minimum, verify the homepage, `/writing/`, `/papers/`, and one post with a table of contents and footnotes.

## Adding Research Papers

Public research is maintained manually in `content/papers/_index.md` using the `paper` shortcode:

```markdown
{{< paper
  title="Paper Title"
  authors="Author One, Author Two"
  date="Month D, YYYY"
  url="https://arxiv.org/abs/0000.00000"
>}}
One-sentence description of the contribution.
{{< /paper >}}
```

Add new papers at the top so the Research page remains reverse chronological.

The `featured="true"` shortcode argument gives a paper the visual “Latest research” treatment on the Research page:

```markdown
{{< paper ... featured="true" >}}
```

There should normally be exactly one featured research card. Move `featured="true"` from the previous paper when promoting a new one.

Do not publish or highlight work that is still private, embargoed, or in peer review unless Shane explicitly asks for it to appear on the site. An arXiv link alone is not approval to feature a paper.

## Highlighting Research on the Homepage

The homepage highlight is configured separately under `params.latestPaper` in `hugo.yaml`:

```yaml
params:
  latestPaper:
    title: "Paper Title"
    date: "Month YYYY"
    url: "https://arxiv.org/abs/0000.00000"
    label: "Latest research"
    summary: "A concise, plain-language description for the homepage."
```

Updating the Research page does not automatically update the homepage. When Shane asks to highlight a paper, update both `content/papers/_index.md` and `params.latestPaper`, then verify that the same paper appears in both places.

To list a paper without promoting it, add it to `content/papers/_index.md` without `featured="true"` and leave `params.latestPaper` unchanged.

## Adding Writing

Create posts under `content/writing/` with Hugo frontmatter. Preserve the established fields:

```yaml
---
title: "Post Title"
date: 2026-01-01T00:00:00Z
author: "Shane Caldwell"
categories: ["llms", "evals"]
tags: ["llms", "evals", "research"]
description: "Search and social description."
summary: "Short archive and homepage summary."
ShowToc: true
TocOpen: false
draft: false
---
```

The homepage automatically displays the four newest pages from the `writing` section. The writing archive groups posts by year and exposes commonly used topic taxonomies.

Standard Markdown footnotes are supported and receive custom desktop and mobile treatment:

```markdown
A claim worth qualifying.[^1]

[^1]: The qualification, citation, joke, or digression.
```

Keep the existing article content, metadata, images, footnotes, code blocks, and URLs intact during visual or template work.

## Deployment

`.github/workflows/hugo.yml` builds and deploys the site to GitHub Pages on pushes to `main`. It uses Hugo Extended 0.146.2 and publishes the generated `public/` artifact.

Do not remove `static/CNAME`; it ensures the generated deployment retains the `hackbot.dad` domain.
