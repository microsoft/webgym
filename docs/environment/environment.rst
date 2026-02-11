RL Pipeline Env
===============

This guide covers setting up the Python environment for the WebGym RL training pipeline.

Prerequisites
-------------

* Python 3.10+
* CUDA-compatible GPU (for training and vLLM inference)
* Conda or virtualenv (recommended)
* rsync (for fast parallel checkpoint copying)

Create Environment
------------------

.. code-block:: bash

   # Create a new conda environment
   conda create -n webgym python=3.10
   conda activate webgym

Install Dependencies
--------------------

1. **Install WebGym with training dependencies:**

   .. code-block:: bash

      pip install -e ".[train]"

   This installs all packages needed for the RL training pipeline, including
   PyTorch, transformers, vLLM, WandB, and other ML dependencies.

   .. tip::

      If you also need the rollout server (OmniBoxes) on the same machine,
      install everything at once:

      .. code-block:: bash

         pip install -e ".[all]"

2. **Clone and install LLaMA-Factory with DeepSpeed:**

   LLaMA-Factory requires a specific DeepSpeed version for compatibility.
   Clone the repository and install with all required extras:

   .. code-block:: bash

      # Clone LLaMA-Factory and pin to a known compatible commit
      git clone https://github.com/hiyouga/LLaMA-Factory.git
      cd LLaMA-Factory && git checkout 8c74dca76a813129c175489c85bf50e2c614091f && cd ..

      # Install with extras
      pip install -e "LLaMA-Factory/[metrics,deepspeed,transformers]" --no-build-isolation

   This installs:

   * ``metrics`` - Training metrics and evaluation
   * ``deepspeed`` - Distributed training with DeepSpeed (version constrained for compatibility)
   * ``transformers`` - HuggingFace transformers integration

Verify Installation
-------------------

.. code-block:: bash

   # Check key packages
   python -c "import webgym; print('WebGym OK')"
   python -c "import llamafactory; print('LLaMA-Factory OK')"
   python -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"
   python -c "import vllm; print('vLLM OK')"

Environment Variables
---------------------

Set the following environment variables before running:

.. code-block:: bash

   # HuggingFace token (required for model downloads)
   export HF_TOKEN="your-huggingface-token"

   # WandB API key (required for logging)
   export WANDB_API_KEY="your-wandb-api-key"

   # CPU cluster token (required for browser instances)
   export CPU_CLUSTER_TOKEN="your-cluster-token"

   # Optional: Gemini API key for evaluation
   export GEMINI_API_KEY="your-gemini-api-key"

Troubleshooting
---------------

**DeepSpeed version conflict:**

If you see an error like ``deepspeed>=0.10.0,<=0.16.9 is required``, ensure you
installed LLaMA-Factory from the cloned repository as shown above, not from PyPI.

**Qwen3-VL processor not found:**

If you see ``ValueError: Processor was not found`` when training with Qwen3-VL models,
ensure you have transformers >= 4.57.1 installed:

.. code-block:: bash

   pip install "transformers==4.57.1"

This version includes ``Qwen3VLProcessor`` which is required for vision-language training.

**vLLM server errors (HTTP 500):**

If you see ``vLLM server returned status 500`` errors during rollout, the vLLM server
is overloaded. Reduce ``env_config.server_size`` (e.g., from 112 to 64-96).
See :ref:`vllm-server-errors` for detailed guidance.
