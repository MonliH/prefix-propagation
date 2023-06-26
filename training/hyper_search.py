import functools
from copy import deepcopy
import math
import importlib
import logging
from transformers.trainer_utils import BestRun, PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import ProgressCallback
from transformers.integrations import TensorBoardCallback, IntervalStrategy, is_datasets_available
import os
import sys

logger = logging.getLogger(__name__)

def run_hp_search_ray(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import ray

    def _objective(trial, local_trainer, checkpoint_dir=None):
        try:
            from transformers.utils.notebook import NotebookProgressCallback

            if local_trainer.pop_callback(NotebookProgressCallback):
                local_trainer.add_callback(ProgressCallback)
        except ModuleNotFoundError:
            pass

        checkpoint = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    checkpoint = os.path.join(checkpoint_dir, subdir)
        local_trainer.objective = None

        tidx = trial["__trial_index__"] + 1
        args = deepcopy(local_trainer.args)
        args.seed = tidx
        local_trainer.args = args
        local_trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(local_trainer, "objective", None) is None:
            metrics = local_trainer.evaluate()
            local_trainer.objective = local_trainer.compute_objective(metrics)
            local_trainer._tune_save_checkpoint()
            ray.tune.report(objective=local_trainer.objective, **metrics, done=True)

    if not trainer._memory_tracker.skip_memory_metrics:
        from transformers.trainer_utils import TrainerMemoryTracker

        logger.warning(
            "Memory tracking for your Trainer is currently "
            "enabled. Automatically disabling the memory tracker "
            "since the memory tracker is not serializable."
        )
        trainer._memory_tracker = TrainerMemoryTracker(skip_memory_metrics=True)

    # The model and TensorBoard writer do not pickle so we have to remove them (if they exists)
    # while doing the ray hp search.
    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    trainer.model = None

    # Setup default `resources_per_trial`.
    if "resources_per_trial" not in kwargs:
        # Default to 1 CPU and 1 GPU (if applicable) per trial.
        kwargs["resources_per_trial"] = {"cpu": 1}
        if trainer.args.n_gpu > 0:
            kwargs["resources_per_trial"]["gpu"] = 1
        resource_msg = "1 CPU" + (" and 1 GPU" if trainer.args.n_gpu > 0 else "")
        logger.info(
            "No `resources_per_trial` arg was passed into "
            "`hyperparameter_search`. Setting it to a default value "
            f"of {resource_msg} for each trial."
        )
    # Make sure each trainer only uses GPUs that were allocated per trial.
    gpus_per_trial = kwargs["resources_per_trial"].get("gpu", 0)
    trainer.args._n_gpu = gpus_per_trial

    # Setup default `progress_reporter`.
    if "progress_reporter" not in kwargs:
        from ray.tune import CLIReporter

        kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])
    if "keep_checkpoints_num" in kwargs and kwargs["keep_checkpoints_num"] > 0:
        # `keep_checkpoints_num=0` would disabled checkpointing
        trainer.use_tune_checkpoints = True
        if kwargs["keep_checkpoints_num"] > 1:
            logger.warning(
                f"Currently keeping {kwargs['keep_checkpoints_num']} checkpoints for each trial. "
                "Checkpoints are usually huge, "
                "consider setting `keep_checkpoints_num=1`."
            )
    if "scheduler" in kwargs:
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule, PopulationBasedTraining

        # Check if checkpointing is enabled for PopulationBasedTraining
        if isinstance(kwargs["scheduler"], PopulationBasedTraining):
            if not trainer.use_tune_checkpoints:
                logger.warning(
                    "You are using PopulationBasedTraining but you haven't enabled checkpointing. "
                    "This means your trials will train from scratch everytime they are exploiting "
                    "new configurations. Consider enabling checkpointing by passing "
                    "`keep_checkpoints_num=1` as an additional argument to `Trainer.hyperparameter_search`."
                )

        # Check for `do_eval` and `eval_during_training` for schedulers that require intermediate reporting.
        if isinstance(
            kwargs["scheduler"], (ASHAScheduler, MedianStoppingRule, HyperBandForBOHB, PopulationBasedTraining)
        ) and (not trainer.args.do_eval or trainer.args.evaluation_strategy == IntervalStrategy.NO):
            raise RuntimeError(
                "You are using {cls} as a scheduler but you haven't enabled evaluation during training. "
                "This means your trials will not report intermediate results to Ray Tune, and "
                "can thus not be stopped early or used to exploit other trials parameters. "
                "If this is what you want, do not use {cls}. If you would like to use {cls}, "
                "make sure you pass `do_eval=True` and `evaluation_strategy='steps'` in the "
                "Trainer `args`.".format(cls=type(kwargs["scheduler"]).__name__)
            )

    trainable = ray.tune.with_parameters(_objective, local_trainer=trainer)

    @functools.wraps(trainable)
    def dynamic_modules_import_trainable(*args, **kwargs):
        """
        Wrapper around `tune.with_parameters` to ensure datasets_modules are loaded on each Actor.

        Without this, an ImportError will be thrown. See https://github.com/huggingface/transformers/issues/11565.

        Assumes that `_objective`, defined above, is a function.
        """
        if is_datasets_available():
            import datasets.load

            dynamic_modules_path = os.path.join(datasets.load.init_dynamic_modules(), "__init__.py")
            # load dynamic_modules from path
            spec = importlib.util.spec_from_file_location("datasets_modules", dynamic_modules_path)
            datasets_modules = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = datasets_modules
            spec.loader.exec_module(datasets_modules)
        return trainable(*args, **kwargs)

    # special attr set by tune.with_parameters
    if hasattr(trainable, "__mixins__"):
        dynamic_modules_import_trainable.__mixins__ = trainable.__mixins__

    analysis = ray.tune.run(
        dynamic_modules_import_trainable,
        config=trainer.hp_space(None),
        num_samples=n_trials,
        **kwargs,
    )
    best_trial = analysis.get_best_trial(metric="objective", mode=direction[:3], scope=trainer.args.ray_scope)
    best_run = BestRun(best_trial.trial_id, best_trial.last_result["objective"], best_trial.config)
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)
    return best_run
