import torch
import evaluate
from datasets import DatasetDict
from markdownify import markdownify as md
import datasets
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NewsgroupsDataset:
    def __init__(
        self, tokenizer: AutoTokenizer, data_args, training_args, rename_to_text=False, use_wandb=True
    ) -> None:
        super().__init__()
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
        data_name = data_args.dataset_name
        assert data_name in ["newsgroups"]
        raw_datasets = load_dataset("SetFit/20_newsgroups")

        # labels
        self.label_list = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
        self.num_labels = len(self.label_list)

        raw_datasets["test"] = raw_datasets["test"].cast_column("label", datasets.ClassLabel(num_classes=self.num_labels, names=self.label_list))
        raw_datasets["train"] = raw_datasets["train"].cast_column("label", datasets.ClassLabel(num_classes=self.num_labels, names=self.label_list))

        split = raw_datasets["test"].train_test_split(test_size=0.5, train_size=0.5, stratify_by_column="label", seed=10)
        raw_datasets["test"] = split["test"]
        raw_datasets["validation"] = split["train"]
        print(raw_datasets)

        self.tokenizer = tokenizer
        self.data_args = data_args

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {id: label for label, id in self.label2id.items()}

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            num_proc=16,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        if rename_to_text:
            raw_datasets = raw_datasets.rename_column("input_ids")

        if training_args.do_train:
            self.train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(
                    range(data_args.max_train_samples)
                )

        if training_args.do_eval:
            self.eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(
                    range(data_args.max_eval_samples)
                )

        if (
            training_args.do_predict
            or data_args.dataset_name is not None
            or data_args.test_file is not None
        ):
            self.predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(
                    range(data_args.max_predict_samples)
                )

        self.metric = load_metric("f1")
        self.ece = evaluate.load("jordyvl/ece")
        self.recall = evaluate.load("recall")
        self.prec = evaluate.load("precision")

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(
                tokenizer, pad_to_multiple_of=8
            )

    def preprocess_function(self, examples):
        # Tokenize the texts
        result = self.tokenizer(
            examples["text"],
            padding=self.padding,
            max_length=self.max_seq_length,
            truncation=True,
        )

        return result

    def compute_metrics(self, p: EvalPrediction):
        _preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(_preds, axis=1)
        if self.data_args.dataset_name is not None:
            result = self.metric.compute(
                predictions=preds, references=p.label_ids, average="micro"
            )
            if self.use_wandb:
                wandb.log(
                    {
                        "conf_mat": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=p.label_ids,
                            preds=preds,
                            class_names=self.label_list,
                        )
                    }
                )

            result["accuracy"] = (preds == p.label_ids).astype(np.float32).mean().item()
            result["recall"] = self.recall.compute(
                predictions=preds, references=p.label_ids, average="macro"
            )["recall"]
            result["precision"] = self.prec.compute(
                predictions=preds, references=p.label_ids, average="macro"
            )["precision"]
            # result["ece"] = self.ece.compute(references=p.label_ids, predictions=torch.nn.functional.softmax(torch.tensor(_preds), dim=1))["ECE"]
            # print(result["ece"])
            return result
        # elif self.is_regression:
        #     return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
