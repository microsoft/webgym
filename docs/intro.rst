Getting Started
===============

To get started with WebGym, you need to prepare two types of machines:

- **CPU machines** for hosting rollout servers (OmniBoxes)
- **GPU machines** for hosting the RL training pipeline

If your GPU machine contains >128 CPUs and >256GB CPU RAM, it is also possible to host CPU services on these GPU machines locally.

First, set up the environments on both machine types. Refer to :doc:`environment/environment_omnibox` for rollout server environment setup and :doc:`environment/environment` for RL pipeline environment setup.

Once the environments are ready, host the rollout server and make sure it is properly tested. Refer to :doc:`server/quickstart_server` for a quickstart guide and :doc:`server/rollout_server` for the full architecture and API details.

Next, launch the training script on the GPU machine. Refer to :doc:`scripts/run_script` for the main entry script and :doc:`scripts/configs` for configuration options.

After RL training completes, analyze the collected trajectories to evaluate performance. Refer to the :doc:`Analysis Tools <analysis/analysis>` section for the available analysis and visualization tools.

For developers looking to understand or extend the system internals, refer to :doc:`rl_pipeline/rl_pipeline` for details on RL pipeline components and development.
