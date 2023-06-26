import logging
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from training.trainer_base import BaseTrainer
from arguments import get_args
import wandb
import os

os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_DISABLED"] = "0"

# Disable INFO logs from longformer; there is some very verbose log telling us about global
# attention on the [CLS] token, which is fine.
logging.getLogger("transformers.models.longformer.modeling_longformer").setLevel(logging.WARNING)

args = get_args()
model_args, data_args, training_args, qa_args = args
set_seed(training_args.seed)

# Set dataset here
# from tasks.arxiv.dataset import ArxivDataset
from tasks.superglue.dataset import SuperGlueDataset

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
# tokenizer = construct_tokenizer(tokenizer)
dataset = SuperGlueDataset(tokenizer, data_args, training_args)

wandb.init(
    project="boolq-kernel",
    entity="queens-law",
    name=os.environ["WANDB_NAME"],
)

config = AutoConfig.from_pretrained(
    model_args.model_name_or_path,
    num_labels=dataset.num_labels,
    hidden_dropout_prob=model_args.hidden_dropout_prob,
)

model= AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    config=config,
)

# Initialize our Trainer
arguments = TrainingArguments(
    do_train=True,
    do_eval=True,
    fp16=training_args.fp16,
    learning_rate=training_args.learning_rate,
    per_device_train_batch_size=training_args.per_device_train_batch_size,
    per_device_eval_batch_size=training_args.per_device_eval_batch_size,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    num_train_epochs=training_args.num_train_epochs,
    output_dir=training_args.output_dir,
    seed=training_args.seed,
    save_strategy=training_args.save_strategy,
    save_total_limit=training_args.save_total_limit,
    save_steps=training_args.save_steps,
    evaluation_strategy=training_args.evaluation_strategy,
    eval_steps=training_args.eval_steps,
    report_to="wandb",
    gradient_checkpointing=training_args.gradient_checkpointing,
    logging_steps=training_args.logging_steps,
)

trainer = BaseTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset.train_dataset if training_args.do_train else None,
    eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=dataset.data_collator,
    compute_metrics=dataset.compute_metrics,
    test_key=dataset.test_key,
)

if data_args.early_stopping_patience >= 0:
    trainer.add_callback(EarlyStoppingCallback(
        early_stopping_patience=data_args.early_stopping_patience))

train_result = trainer.train()

metrics = train_result.metrics

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

trainer.log_best_metrics()
trainer.save_model()
