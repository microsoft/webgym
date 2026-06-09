import dataclasses
import json
import os
import pickle
from typing import TYPE_CHECKING

import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import RecoverConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.io_struct import FinetuneSpec, SaveLoadMeta, StepInfo, WeightUpdateMeta
from areal.infra import TrainController
from areal.utils import logging
from areal.utils.evaluator import Evaluator
from areal.utils.saver import Saver

if TYPE_CHECKING:
    from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("Recover")


class InValidRecoverInfo(Exception):
    pass


@dataclasses.dataclass
class RecoverInfo:
    # Last step info is the counter of the saved checkpoint.
    # Recover will start from the next iteration, obtained by `last_step_info.next()`.
    last_step_info: StepInfo

    saver_info: dict
    evaluator_info: dict
    stats_logger_info: dict
    dataloader_info: dict | list[dict]
    checkpoint_info: dict

    def dump(self, dump_dir: str):
        # Dumps the recover info to multiple files in `dump_dir`:
        # 1. step_info.json: contains the recover info
        # 2. *_info.json or *_info.pkl: contains other informantion required for recover.

        if dist.is_initialized():
            # Since dataloader state is different across distributed ranks,
            # we need to all gather the dataloader state from all ranks.
            # In this situation, saved dataloader_info is a list of states from all ranks.
            dataloader_info = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(dataloader_info, self.dataloader_info)

            # To avoid contention, do not dump on multiple ranks
            if dist.get_rank() != 0:
                return
        else:
            dataloader_info = self.dataloader_info

        os.makedirs(dump_dir, exist_ok=True)
        step_info_path = os.path.join(dump_dir, "step_info.json")
        with open(step_info_path, "w") as f:
            json.dump(dataclasses.asdict(self.last_step_info), f, indent=4)

        saver_info_path = os.path.join(dump_dir, "saver_info.json")
        with open(saver_info_path, "w") as f:
            json.dump(self.saver_info, f, indent=4)

        evaluator_info_path = os.path.join(dump_dir, "evaluator_info.json")
        with open(evaluator_info_path, "w") as f:
            json.dump(self.evaluator_info, f, indent=4)

        stats_logger_info_path = os.path.join(dump_dir, "stats_logger_info.json")
        with open(stats_logger_info_path, "w") as f:
            json.dump(self.stats_logger_info, f, indent=4)

        checkpoint_info_path = os.path.join(dump_dir, "checkpoint_info.json")
        with open(checkpoint_info_path, "w") as f:
            json.dump(self.checkpoint_info, f, indent=4)

        dataloader_info_path = os.path.join(dump_dir, "dataloader_info.pkl")
        with open(dataloader_info_path, "wb") as f:
            pickle.dump(dataloader_info, f)

    @classmethod
    def load(cls, load_dir: str):
        # Loads the recover info from multiple files in `load_dir`:
        if not os.path.exists(load_dir):
            raise FileNotFoundError(
                f"Recover info directory {load_dir} does not exist."
            )

        try:
            step_info_path = os.path.join(load_dir, "step_info.json")
            with open(step_info_path) as f:
                step_info_dict = json.load(f)
                last_step_info = StepInfo(**step_info_dict)

            evaluator_info_path = os.path.join(load_dir, "evaluator_info.json")
            with open(evaluator_info_path) as f:
                evaluator_info = json.load(f)

            saver_info_path = os.path.join(load_dir, "saver_info.json")
            with open(saver_info_path) as f:
                saver_info = json.load(f)

            stats_logger_info_path = os.path.join(load_dir, "stats_logger_info.json")
            with open(stats_logger_info_path) as f:
                stats_logger_info = json.load(f)

            checkpoint_info_path = os.path.join(load_dir, "checkpoint_info.json")
            with open(checkpoint_info_path) as f:
                checkpoint_info = json.load(f)

            dataloader_info_path = os.path.join(load_dir, "dataloader_info.pkl")
            with open(dataloader_info_path, "rb") as f:
                dataloader_info = pickle.load(f)
                if isinstance(dataloader_info, list):
                    # If dataloader_info a list, it means it is saved from a distributed run.
                    if dist.is_initialized():
                        # Loading dataloader states in a distributed context.
                        assert dist.get_world_size() == len(dataloader_info), (
                            f"Dataloader info list length {len(dataloader_info)} does not match "
                            f"the world size {dist.get_world_size()}."
                        )
                        dataloader_info = dataloader_info[dist.get_rank()]

            return cls(
                last_step_info=last_step_info,
                saver_info=saver_info,
                evaluator_info=evaluator_info,
                stats_logger_info=stats_logger_info,
                dataloader_info=dataloader_info,
                checkpoint_info=checkpoint_info,
            )
        except Exception as e:
            logger.error(f"Failed to load recover info from {load_dir}: {e}")
            raise InValidRecoverInfo(f"Invalid recover info in {load_dir}") from e


class RecoverHandler:
    def __init__(self, config: RecoverConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_spec = ft_spec
        self.last_step_info = StepInfo(
            epoch=-1,
            epoch_step=-1,
            global_step=-1,
            steps_per_epoch=ft_spec.steps_per_epoch,
        )

    @staticmethod
    def recover_info_path(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
    ):
        return os.path.join(
            Saver.get_save_root(experiment_name, trial_name, fileroot),
            "recover_info",
        )

    def dump(
        self,
        engine: TrainEngine | dict[str, TrainEngine],
        step_info: StepInfo,
        saver: Saver,
        evaluator: Evaluator,
        stats_logger: "StatsLogger",
        dataloader: StatefulDataLoader,
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
    ):
        if isinstance(engine, TrainEngine):
            engine = {"default": engine}
        for name, engine_ in engine.items():
            self._save_checkpoint(
                engine_,
                name=name,
                tokenizer=tokenizer,
                processor=processor,
                base_model_path=base_model_path,
            )

        self.last_step_info = step_info
        recover_info = RecoverInfo(
            last_step_info=self.last_step_info,
            saver_info=saver.state_dict(),
            evaluator_info=evaluator.state_dict(),
            stats_logger_info=stats_logger.state_dict(),
            dataloader_info=dataloader.state_dict(),
            checkpoint_info={},
        )

        recover_info_path = self.recover_info_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
        )
        recover_info.dump(recover_info_path)

    def load(
        self,
        engine: TrainEngine | dict[str, TrainEngine] | TrainController,
        saver: Saver,
        evaluator: Evaluator,
        stats_logger: "StatsLogger",
        dataloader: StatefulDataLoader,
        inference_engine: InferenceEngine | None = None,
        weight_update_meta: WeightUpdateMeta | None = None,
        inference_engine_update_from: str = "default",
        load_only: str = "all",
    ) -> RecoverInfo | None:
        if inference_engine is not None and weight_update_meta is None:
            raise ValueError("Weight update meta is required for recovery.")

        if isinstance(engine, (TrainEngine, TrainController)):
            engine = {"default": engine}

        recover_info_path = self.recover_info_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
        )
        logger.info(f"Loading recover info from {recover_info_path}")
        try:
            recover_info: RecoverInfo = RecoverInfo.load(recover_info_path)
            logger.info(f"Recovering from {recover_info.last_step_info.next()}.")

            # Full recover: restore all state (step, saver, dataloader, etc.)
            # Partial recover (load_only != "all"): only load specified engines,
            # start from step 0 with fresh training state.
            if load_only in ("all", "actor"):
                saver.load_state_dict(recover_info.saver_info)
                evaluator.load_state_dict(recover_info.evaluator_info)
                stats_logger.load_state_dict(recover_info.stats_logger_info)
                dataloader.load_state_dict(recover_info.dataloader_info)

            if load_only == "none":
                logger.info("Skipping all recover loads (load_only='none')")
            else:
                # global_step counted from 0; #scheduler.step() calls before save = global_step + 1.
                sched_catch_up = recover_info.last_step_info.global_step + 1
                for name, engine_ in engine.items():
                    if load_only != "all":
                        allowed = {"default"} if load_only == "actor" else {load_only}
                        if name not in allowed:
                            logger.info(
                                "Skipping recover load for '%s' (load_only='%s')",
                                name, load_only,
                            )
                            continue
                    try:
                        self._load_checkpoint(engine_, name=name)
                    except (EOFError, RuntimeError) as e:
                        logger.warning(
                            "Failed to load recover checkpoint for '%s': %s. "
                            "Skipping (engine will use initial weights).",
                            name, e,
                        )
                        continue
                    # DCP restored optimizer.param_groups[i]["lr"], but the
                    # scheduler is still at its post-init last_epoch=0. Replay
                    # step() so the current schedule lambda re-derives
                    # (last_epoch, lr) for the resumed iteration.
                    if sched_catch_up > 0:
                        logger.info(
                            "Catching up LR scheduler for '%s' by %d steps",
                            name, sched_catch_up,
                        )
                        engine_.catch_up_lr_scheduler(sched_catch_up)

            if load_only in ("all", "actor"):
                global_step = recover_info.last_step_info.global_step
            else:
                global_step = 0
                # Reset step_info so train() sees step 0
                recover_info.last_step_info = StepInfo(
                    global_step=-1,
                    epoch=0,
                    epoch_step=-1,
                    steps_per_epoch=recover_info.last_step_info.steps_per_epoch,
                )
                logger.info(
                    "Partial recover (load_only='%s'): starting from step 0",
                    load_only,
                )

            if inference_engine is not None:
                update_engine = engine[inference_engine_update_from]
                update_engine.connect_engine(inference_engine, weight_update_meta)
                # update inference engine weights
                inference_engine.pause()
                update_engine.update_weights(weight_update_meta)
                inference_engine.resume()
                update_engine.set_version(global_step + 1)
                inference_engine.set_version(global_step + 1)
                # Seed the rollout's staleness manager so its capacity formula
                # behaves like steady state from step 1 of the resumed run.
                # Without this the manager's ``accepted`` counter starts at 0
                # while ``current_version`` is already (global_step + 1),
                # leaving ~(max_staleness + version + 1) * batch slots open
                # and admitting a flood of stale rollouts.
                seed_fn = getattr(inference_engine, "seed_staleness_for_resume", None)
                if seed_fn is not None:
                    seed_fn(global_step + 1)
            return recover_info
        except (FileNotFoundError, InValidRecoverInfo):
            logger.warning(
                f"Resume info not found at {recover_info_path}. "
                f"This should not be a resumed experiment!"
            )

    def _save_checkpoint(
        self,
        engine: TrainEngine,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
    ):
        path = Saver.get_recover_checkpoint_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
            name=name,
        )
        weight_format = "dcp"
        with_optim = True
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=tokenizer,
            processor=processor,
            base_model_path=base_model_path,
        )
        engine.save(meta)
        logger.info(f"Saved recover checkpoint to {path}")

    def _load_checkpoint(
        self,
        engine: TrainEngine | TrainController,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        base_model_path: str | None = None,
    ):
        path = Saver.get_recover_checkpoint_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
            name=name,
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path {path} does not exist.")
        weight_format = "dcp"
        with_optim = True
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=None,
            processor=None,
            base_model_path=None,
        )
        engine.load(meta)


def check_if_auto_recover(config: RecoverConfig) -> bool:
    # This method is called by check_if_recover to check if the experiment should
    # recover from a previous run when recovery is enabled ("on" or "auto" mode).
    experiment_name = config.experiment_name
    trial_name = config.trial_name
    fileroot = config.fileroot
    recover_info_path = RecoverHandler.recover_info_path(
        experiment_name, trial_name, fileroot
    )
    logger.info(f"Searching for recover info file in {recover_info_path}.")
    if os.path.exists(str(recover_info_path)):
        try:
            info = RecoverInfo.load(recover_info_path)
        except Exception as e:
            logger.warning(f"Failed to load recover info from {recover_info_path}: {e}")
            return False
        if info.last_step_info.epoch < 0:
            msg = (
                f"Recover checkpoint is not valid. "
                f"Expected last_step_info.epoch >= 0, "
                f"but found {info.last_step_info.epoch}"
            )
            logger.warning(msg)
            return False

        save_root = Saver.get_save_root(experiment_name, trial_name, fileroot)
        for name in os.listdir(save_root):
            if not os.path.isdir(os.path.join(save_root, name)):
                continue
            path = Saver.get_recover_checkpoint_path(
                experiment_name, trial_name, fileroot, name=name
            )
            if not os.path.exists(path):
                logger.warning(f"Recover checkpoint for model {name} does not exist.")
                return False
        return True
    logger.warning(f"Recover info not found at: {recover_info_path}")
    return False


def check_if_recover(config: RecoverConfig, _run_id: int) -> bool:
    """Check if the experiment should recover from a previous run.

    Always checks for valid recover info and checkpoints.

    Args:
        config: Recovery configuration.
        _run_id: Unused. Kept for API compatibility.

    Returns:
        True if the experiment should recover from a previous run.
    """
    return check_if_auto_recover(config)
