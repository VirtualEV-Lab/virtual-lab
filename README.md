# Virtual Lab

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/virtual-lab)](https://badge.fury.io/py/virtual-lab)
[![PyPI version](https://badge.fury.io/py/virtual-lab.svg)](https://badge.fury.io/py/virtual-lab)
[![Downloads](https://pepy.tech/badge/virtual-lab)](https://pepy.tech/project/virtual-lab)
[![license](https://img.shields.io/github/license/zou-group/virtual-lab.svg)](https://github.com/zou-group/virtual-lab/blob/main/LICENSE.txt)

![Virtual Lab](https://github.com/zou-group/virtual-lab/raw/main/images/virtual_lab_architecture.png)

The **Virtual Lab** is an AI-human collaboration for science research. In the Virtual Lab, a human researcher works with a team of large language model (LLM) **agents** to perform scientific research. Interaction between the human researcher and the LLM agents occurs via a series of **team meetings**, where all the LLM agents discuss a scientific agenda posed by the human researcher, and **individual meetings**, where the human researcher interacts with a single LLM agent to solve a particular scientific task.

Please see our paper [The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies](https://www.nature.com/articles/s41586-025-09442-9) for more details on the Virtual Lab and an application to nanobody design for SARS-CoV-2.

If you use the Virtual Lab, please cite our work as follows:

Swanson, K., Wu, W., Bulaong, N.L. et al. The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies. *Nature* (2025). https://doi.org/10.1038/s41586-025-09442-9


## Virtual Lab for nanobody design

As a real-world demonstration, we applied the Virtual Lab to design nanobodies for one of the latest variants of SARS-CoV-2 (see [nanobody_design](https://github.com/zou-group/virtual-lab/tree/main/nanobody_design)). The Virtual Lab built a computational pipeline consisting of [ESM](https://www.science.org/doi/10.1126/science.ade2574), [AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2), and [Rosetta](https://rosettacommons.org/software/) and used it to design 92 nanobodies that were experimentally validated.

Please see the notebook [nanobody_design/run_nanobody_design.ipynb](https://github.com/zou-group/virtual-lab/blob/main/nanobody_design/run_nanobody_design.ipynb) for an example of how to use the Virtual Lab to create agents and run team and individual meetings.


## Installation

The Virtual Lab can be installed using pip or by cloning the repo and installing the required packages. Installation should only take a couple of minutes.

Optionally, first create a conda environment.

```bash
conda create -y -n virtual_lab python=3.14
conda activate virtual_lab
```

The Virtual Lab can be installed via pip.

```bash
pip install virtual-lab
```

To install the latest version of the Virtual Lab locally, clone the repo and then install the package.

```bash
git clone https://github.com/zou-group/virtual_lab.git
cd virtual_lab
pip install -e .
```


## API key

The Virtual Lab uses GPT-5.2 from OpenAI by default. Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` and replace the placeholder with your actual key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

The `.env` file is automatically loaded when the package is imported via `python-dotenv`. The `.env` file is already included in `.gitignore` to prevent accidental commits of your API key.

Alternatively, you can still export the environment variable directly (e.g., `export OPENAI_API_KEY=<your_key>` in your shell). Environment variables set in the shell take precedence over values in `.env`.

### DeepSeek

To call [DeepSeek](https://api-docs.deepseek.com/) instead of OpenAI, set `LLM_PROVIDER=deepseek` and a DeepSeek API key. The same OpenAI Python client is used with DeepSeek's OpenAI-compatible endpoint (`https://api.deepseek.com`).

```
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

Agent definitions in this repo name OpenAI models (`gpt-5.2`, `gpt-4o-2024-08-06`, and so on). With `LLM_PROVIDER=deepseek`, those names are sent as `deepseek-flash` (DeepSeek-V4.1-Flash). Set `DEEPSEEK_MODEL=deepseek-v4-pro` to use DeepSeek-V4-Pro instead. An agent whose `model` already starts with `deepseek` keeps that id.

Optional settings:

```
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_THINKING=disabled
```

`DEEPSEEK_THINKING` defaults to `disabled` so the meeting `temperature` is applied. Set it to `enabled` for DeepSeek's chain-of-thought mode. Thinking mode ignores `temperature`.
