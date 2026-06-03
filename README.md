# Lectures

[![Docs](https://github.com/jejjohnson/lectures/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/lectures)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)

> A home for my lecture material — deep, reproducible **reference notes**, curated
> reveal.js **slide decks**, and hands-on **labs**. The first course is *Probabilistic
> ML for Finance* (dependence, Gaussianization, uncertainty, fair learning); more will
> be added as their own courses.

## Two layers

| Layer | Where | What |
|---|---|---|
| **Topics** | `topics/` | Evergreen deep reference. Full derivations, executable code, reproducible figures. Reused across courses. |
| **Courses** | `courses/` | Curated playlists. Lean, ELI5 reveal.js slides that pick the key pieces from the topics, plus live-coding demos. |

Authoring rule: write the deep material + figures **once** in `topics/`; each course is a
curated selection that references them. Future courses = a new `courses/<id>/` folder.

## Structure

```
lectures/
├── _quarto.yml              # Quarto project: builds the website AND the slides
├── index.qmd                # landing page
├── topics/                  # ① deep, evergreen content (Quarto .qmd + executable code)
│   ├── dependence/          #    correlation · mutual information · HSIC · causality
│   ├── gaussianization/     #    rbig · gauss_flows
│   ├── uncertainty/         #    noise propagation · predicted noise + ensembles
│   └── fairness/            #    keras-fairkl (fair kernel learning)
├── courses/
│   └── 2025-finance-msc/    # ② curated course
│       ├── index.qmd        #    syllabus + 10h schedule
│       ├── slides/          #    reveal.js decks (format: revealjs)
│       └── demos/           #    Jupytext .py live-coding notebooks
├── src/lectures/            # shared helpers: plotting style + toy datasets
├── styles/                  # site + slide SCSS themes
└── references.bib
```

## Quick start

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # if you don't have pixi
pixi install

pixi run -e docs preview     # live-reload site + slides at localhost:4848
pixi run -e docs render      # build everything -> _site/
```

Render a single deck while iterating:

```bash
pixi run -e docs quarto render courses/2025-finance-msc/slides/01-dependence.qmd
```

## Code demos

The demos build on these libraries, installed (not copied) via the opt-in `demos` env:
[rbig](https://github.com/jejjohnson/rbig),
[gauss_flows](https://github.com/jejjohnson/gauss_flows),
[pyrox](https://github.com/jejjohnson/pyrox),
[keras-fairkl](https://github.com/jejjohnson/keras-fairkl).

```bash
pixi install -e demos        # heavy (jax/torch/tf); kept out of the docs build
pixi run -e demos python courses/2025-finance-msc/demos/01_dependence.py
```

## Building on this

Module 1 (Dependence) is built end-to-end as the worked example: deep notes with
generated figures (`topics/dependence/01-correlation.qmd`) and a matching reveal.js deck
(`courses/2025-finance-msc/slides/01-dependence.qmd`). Modules 2–4 are stubbed with the
same structure, ready to fill from their topic chapters.

Deployed to GitHub Pages on every push to `main` via `.github/workflows/pages.yml`.
