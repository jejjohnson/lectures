# Getting Started

## Prerequisites

Install [Pixi](https://pixi.sh) for environment management:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## Installation

Clone the repository and install the environment:

```bash
git clone https://github.com/jejjohnson/lectures
cd lectures
pixi install
```

## Build the Site

The notes, slide decks, and labs are one Quarto site.

```bash
# Live-reload preview of the site + slides (http://localhost:4848)
pixi run -e docs preview

# Render everything to _site/
pixi run -e docs render
```

## Run the Tests

```bash
pixi run test
```

## JupyterLab

For exploring the library and labs interactively:

```bash
pixi run -e jupyterlab lab
```
