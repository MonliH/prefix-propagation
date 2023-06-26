import logging
from copy import deepcopy

from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from model.utils import get_model, TaskType
from tasks.wikihop.dataset import WikiHopDataset, get_wikihop_tokenizer
from training.trainer_wikihop import WikiHopTrainer

logger = logging.getLogger(__name__)


def get_trainer(args, use_wandb=True):
    model_args, data_args, training_args, _ = args
    tokenizer = get_wikihop_tokenizer(model_args.model_name_or_path)
    dataset = WikiHopDataset(tokenizer, data_args, training_args, use_wandb=use_wandb)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task=data_args.dataset_name,
        revision=model_args.model_revision,
    )
    model_args.additional_non_frozen_embeds = 4
    config.additional_non_frozen_embeds = 4

    model = get_model(model_args, TaskType.WIKI_HOP, config)
    training_args.remove_unused_columns = False

    # Initialize our Trainer
    trainer = WikiHopTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        eval_data_collator=dataset.eval_data_collator,
        train_data_collator=dataset.train_data_collator,
    )

    return trainer, dataset.eval_dataset
