import logging
import torch
import os
import sys
import numpy as np
from typing import Dict
import wandb

import datasets
import transformers
from transformers import set_seed, Trainer, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint

from arguments import get_args

from tasks.utils import *

logger = logging.getLogger(__name__)

def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.log_best_metrics()


def evaluate(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")
        return
    elif isinstance(predict_dataset, dict):
        for dataset_name, d in predict_dataset.items():
            logger.info("*** Predict: %s ***" % dataset_name)
            predictions, labels, metrics = trainer.predict(
                d, metric_key_prefix="predict"
            )
            predictions = np.argmax(predictions, axis=2)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)
            for k, v in metrics.items():
                wandb.run.summary[k] = v
            return metrics
    else:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        )
        predictions = np.argmax(predictions, axis=0)
        print(metrics)
        # for k, v in metrics.items():
        #     wandb.run.summary[k] = v

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        return metrics


def main():
    torch.autograd.set_detect_anomaly(True)

    args = get_args()

    _, data_args, training_args, _ = args

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    dataset_name = data_args.task_name.lower()

    if dataset_name == "superglue":
        assert data_args.dataset_name.lower() in SUPERGLUE_DATASETS
        from tasks.superglue.get_trainer import get_trainer
    elif dataset_name == "glue":
        assert data_args.dataset_name.lower() in GLUE_DATASETS
        from tasks.glue.get_trainer import get_trainer
    elif dataset_name == "hyperpartisan":
        from tasks.hyperpartisan.get_trainer import get_trainer
    elif dataset_name == "arxiv":
        from tasks.arxiv.get_trainer import get_trainer
    elif dataset_name == "wikihop":
        from tasks.wikihop.get_trainer import get_trainer
    elif dataset_name == "newsgroups":
        from tasks.newsgroups.get_trainer import get_trainer
    else:
        print(dataset_name)
        raise NotImplementedError(
            "Task {} is not implemented. Please choose a dataset from: {}".format(
                data_args.task_name, ", ".join(DATASETS)
            )
        )

    set_seed(training_args.seed)

    # disable annoying longformer logs
    use_wandb = False
    logging.getLogger("transformers.models.longformer.modeling_longformer").setLevel(logging.WARNING)
    if "wandb" in training_args.report_to:
        import wandb
        use_wandb = True
        os.environ["WANDB_MODE"] = "online"

        entity, project = os.environ["WANDB_PROJECT_NAME"].split("/")

        wandb.init(
            project=project,
            entity=entity,
            name=os.environ["WANDB_NAME"],
            config={
                "lineage": os.environ["LINEAGE"],
            },
        )

    trainer, predict_dataset = get_trainer(args, use_wandb=use_wandb)

    # Early stopping
    if data_args.early_stopping_patience >= 0:
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=data_args.early_stopping_patience))

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if training_args.do_train:
        train(trainer, training_args.resume_from_checkpoint, last_checkpoint)

    if training_args.do_eval:
        evaluate(trainer)

    if training_args.do_predict:
        predict(trainer, predict_dataset)

if __name__ == "__main__":
    main()