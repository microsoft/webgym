<h1 align="center">AsyncWebRL: Efficient Multi-Step RL for Visual Web Agents</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-red)](https://arxiv.org/abs/2606.05597)
[![Documentation Status](https://app.readthedocs.org/projects/asyncwebrl/badge/?version=latest)](https://asyncwebrl.readthedocs.io/en/latest/)
[![Project Page](https://img.shields.io/badge/🌐%20Project-Page-blue)](https://asyncwebrl-website.github.io/)
[![Code](https://img.shields.io/badge/Code-WebGym%20async-black?logo=github)](https://github.com/microsoft/webgym/tree/async)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

</div>

<div align="center">
  <img src="assets/banner.png" alt="AsyncWebRL" width="480">
</div>

<p align="center">
<a href="https://www.jackgethome.com/"><b>Hao Bai</b></a><sup>1,2</sup>, <a href="https://yangrui2015.github.io/"><b>Rui Yang</b></a><sup>1,2</sup>, <a href="https://chenluye99.github.io/"><b>Chenlu Ye</b></a><sup>1</sup>, <a href="https://www.spencerwhitehead.com/"><b>Spencer Whitehead</b></a><sup>2</sup>,
<br>
<a href="https://aviralkumar2907.github.io/"><b>Aviral Kumar</b></a><sup>3</sup>, <a href="https://tongzhang-ml.org/"><b>Tong Zhang</b></a><sup>1</sup>
</p>

<p align="center">
<sup>1</sup>UIUC &nbsp; <sup>2</sup>Microsoft &nbsp; <sup>3</sup>CMU
</p>

AsyncWebRL trains vision-language web agents with efficient multi-step reinforcement learning. It is built on the [AReaL](https://github.com/inclusionAI/AReaL) async RL framework and sets a new open-source state of the art on the [WebGym](https://github.com/microsoft/webgym) out-of-distribution test split.

Note, for the original WebGym code, please go to the [`webgym` branch](https://github.com/microsoft/webgym/tree/webgym)

## Features

**Asynchronous system — up to 2.9× training-throughput speedup** over the
previously fastest open synchronous pipeline (WebGym):

- **Everlasting rollout pool** — rollout workers stay alive across iteration
  boundaries, so rollout, gradient update, and policy refresh overlap
  continuously with no per-iteration warm-up bubble.
- **Lightweight screenshot handling** — per-step image tensors stay in a
  dedicated in-memory actor; only lightweight references travel over RPC,
  avoiding the shared object-store serialization that bottlenecks WebGym.
- **Decoupled off-policy correction** — a decoupled importance-sampling ratio
  that roughly halves clip-trigger rates under async off-policyness.

**Algorithmic fix — shorter trajectories at the same success rate:**

- Diagnoses the per-trajectory step normalizer $1/|\tau_i|$ in multi-step GRPO
  as the root cause of trajectory- and token-level inefficiency (failures run
  far longer than successes, so it under-weights the negative gradient on
  failed tokens).
- Replacing it with a constant $1/k$ breaks this coupling — trajectories
  contract while aggregate success is preserved, with the largest gains on the
  harder Medium / Hard OOD slices.

**Framework:**

- Built on AReaL — FSDP2 / Megatron training, SGLang / vLLM rollout,
  Qwen3-VL policies.
- Modular and extensible: workflows, engines, rewards, and datasets are
  independent, swappable components.

## Documentation

Everything beyond this page — installation (Docker image or local `uv`),
quickstart, configuration reference, and system / algorithm design — lives in
the docs:

| Topic | Link |
|---|---|
| Installation — verified package versions | [Installation](https://asyncwebrl.readthedocs.io/en/latest/installation.html) |
| Run training, adapt the config to your cluster | [Quickstart](https://asyncwebrl.readthedocs.io/en/latest/quickstart.html) |
| Configuration reference | [Configuration](https://asyncwebrl.readthedocs.io/en/latest/cli_reference.html) |
| Async system design | [Async System](https://asyncwebrl.readthedocs.io/en/latest/async_system.html) |
| Algorithm design | [Algorithm](https://asyncwebrl.readthedocs.io/en/latest/algorithm.html) |

## Citation

```bibtex
@article{bai2026asyncwebrl,
  title     = {AsyncWebRL: Efficient Multi-Step RL for Visual Web Agents},
  author    = {Bai, Hao and Yang, Rui and Ye, Chenlu and Whitehead, Spencer and Kumar, Aviral and Zhang, Tong},
  journal   = {arXiv preprint arXiv:2606.05597},
  year      = {2026}
}
```


## License

The code in this repository is under an [MIT license](LICENSE). Part of our code is based on AReaL, which is under an [Apache 2.0 License](https://github.com/areal-project/AReaL/blob/main/LICENSE).
