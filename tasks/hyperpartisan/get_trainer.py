import logging
from copy import deepcopy

from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from model.utils import get_model, TaskType
from tasks.hyperpartisan.dataset import HyperpartisanDataset
from training.trainer_base import BaseTrainer

logger = logging.getLogger(__name__)


def get_trainer(args, use_wandb=True):
    model_args, data_args, training_args, _ = args

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    dataset = HyperpartisanDataset(tokenizer, data_args, training_args, use_wandb=use_wandb)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=dataset.num_labels,
        label2id=dataset.label2id,
        id2label=dataset.id2label,
        problem_type="single_label_classification",
        finetuning_task=data_args.dataset_name,
        revision=model_args.model_revision,
    )

    model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)

    # Initialize our Trainer
    trainer = BaseTrainer(
        test_key="f1",
        args=training_args,
        model=model,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
    )

    return trainer, dataset.predict_dataset
