# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is being migrated from Quarto to Hugo for better mobile experience and technical blog features. The site focuses on cybersecurity, AI, and technical content.

**Migration goals:**
- Clean mobile reading experience
- Good-looking inline code highlighting
- RSS feed support
- GoatCounter analytics integration
- Footnote support
- GitHub Pages deployment

## Current State (Quarto → Hugo Migration)

**Existing content structure:**
- `posts/` - Blog posts in Quarto Markdown (.qmd)
- `_quarto.yml` - Current Quarto configuration 
- `docs/` - Generated output (will be replaced by Hugo's `public/`)

**Content to preserve:**
- All blog posts, especially technical security content
- Author info: Shane Caldwell
- Post categories and dates
- Images and assets

## Development Commands (Post-Migration)

**Hugo site generation:**
```bash
hugo
```

**Development server:**
```bash
hugo server -D
```

**GitHub Pages deployment:**
Hugo will generate static files to `public/` directory for GitHub Pages deployment.

## Migration Notes

**Content conversion needed:**
- Convert .qmd files to Hugo Markdown (.md)
- Update frontmatter format from Quarto to Hugo
- Ensure footnote syntax compatibility
- Preserve code blocks and syntax highlighting
- Maintain post categories and metadata