Rollout Server Env
==================

This guide covers setting up the Python environment for the OmniBoxes rollout server.
This is a lightweight installation that does **not** require a GPU or ML libraries.

Prerequisites
-------------

* Python 3.10+
* Conda or virtualenv (recommended)

Create Environment
------------------

.. code-block:: bash

   # Create a new conda environment
   conda create -n webgym python=3.10
   conda activate webgym

Install Dependencies
--------------------

1. **Install WebGym with omnibox dependencies:**

   .. code-block:: bash

      pip install -e ".[omnibox]"

   This installs only the packages needed for the OmniBoxes rollout server:
   FastAPI, httpx, Playwright, Redis, Pillow, psutil, and Requests.

   .. tip::

      If you also need the RL training pipeline on the same machine,
      install everything at once:

      .. code-block:: bash

         pip install -e ".[all]"

2. **Install Redis:**

   With sudo (Ubuntu/Debian):

   .. code-block:: bash

      sudo apt-get update
      sudo apt-get install redis-server

   Without sudo (compile from source):

   .. code-block:: bash

      wget https://download.redis.io/redis-stable.tar.gz
      tar -xzvf redis-stable.tar.gz
      cd redis-stable
      make
      export PATH="$(pwd)/src:$PATH"

   To make the PATH change permanent, add it to your ``.bashrc``:

   .. code-block:: bash

      echo 'export PATH="/path/to/redis-stable/src:$PATH"' >> ~/.bashrc
      source ~/.bashrc

   macOS:

   .. code-block:: bash

      brew install redis

   Verify Redis is installed:

   .. code-block:: bash

      redis-cli --version

3. **Install Playwright browsers and system dependencies:**

   .. code-block:: bash

      playwright install chromium
      playwright install-deps chromium

4. **(Optional) Install nginx:**

   The ``--nginx`` flag is experimental and not yet supported.

Verify Installation
-------------------

.. code-block:: bash

   # Check key packages
   python -c "import fastapi; print('FastAPI OK')"
   python -c "import playwright; print('Playwright OK')"
   python -c "import redis; print('Redis OK')"
   redis-cli --version

Launch the Server
-----------------

.. code-block:: bash

   cd omniboxes/deploy

   # Local development (e.g. 128 instances)
   python deploy.py 128

See :doc:`../server/quickstart_server` for full deployment options and API usage.
