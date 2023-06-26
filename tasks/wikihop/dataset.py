# Wikihop model from:
# "Longformer: The Long-Document Transformer", Beltagy et al, 2020: https://arxiv.org/abs/2004.05150
# Code from https://github.com/allenai/longformer/blob/mp/wikihop/scripts/wikihop.py


# Before training, download and prepare the data. The data preparation step takes a few minutes to tokenize and save the data.
# (1) Download data from http://qangaroo.cs.ucl.ac.uk/
# (2) unzip the file `unzip qangaroo_v1.1.zip`.  This creates a directory `qangaroo_v1.1`.
# (3) Prepare the data (tokenize, etc): `python scripts/wikihop.py --prepare-data --data-dir /path/to/qarangoo_v1.1/wikihop`

import evaluate
import random
from itertools import chain
from dataclasses import dataclass
import torch
from datasets.load import load_dataset, load_metric
from datasets.dataset_dict import DatasetDict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_wikihop_tokenizer(tokenizer_name="allenai/longformer-base-4096"):
    additional_tokens = ["[question]", "[/question]", "[ent]", "[/ent]"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    tokenizer.add_tokens(additional_tokens)

    return tokenizer


def normalize_string(s):
    s = s.replace(" .", ".")
    s = s.replace(" ,", ",")
    s = s.replace(" !", "!")
    s = s.replace(" ?", "?")
    s = s.replace("( ", "(")
    s = s.replace(" )", ")")
    s = s.replace(" 's", "'s")
    return " ".join(s.strip().split())


@dataclass
class DataCollatorForWikiHop:
    tokenizer: PreTrainedTokenizerBase
    # Set this to true during training (but not during validation to keep things consistent)
    shuffle_candidates: bool

    def __call__(self, instance):
        assert isinstance(instance, list)
        def one_entry(instance):
            # list of wordpiece tokenized candidates surrounded by [ent] and [/ent]
            candidate_tokens = instance["candidate_tokens"]
            # list of word piece tokenized support documents surrounded by </s> </s>
            supports_tokens = instance["supports_tokens"]
            query_tokens = instance["query_tokens"]
            answer_index = instance["answer_index"]

            n_candidates = len(candidate_tokens)
            sort_order = list(range(n_candidates))

            # concat all the candidate_tokens with <s>: <s> + candidates
            all_candidate_tokens = ["<s>"] + query_tokens

            # candidates
            n_candidates = len(candidate_tokens)
            sort_order = list(range(n_candidates))

            # Shuffle candidates if training
            if self.shuffle_candidates:
                random.shuffle(sort_order)
                new_answer_index = sort_order.index(answer_index)
                answer_index = new_answer_index
            all_candidate_tokens.extend(
                chain.from_iterable([candidate_tokens[k] for k in sort_order])
            )

            # the supports
            n_supports = len(supports_tokens)
            sort_order = list(range(n_supports))
            if self.shuffle_candidates:
                random.shuffle(sort_order)
            all_support_tokens = list(
                chain.from_iterable([supports_tokens[k] for k in sort_order])
            )

            # convert to ids
            candidate_ids = self.tokenizer.convert_tokens_to_ids(all_candidate_tokens)
            support_ids = self.tokenizer.convert_tokens_to_ids(all_support_tokens)

            # get the location of the predicted indices
            predicted_indices = [
                k for k, token in enumerate(all_candidate_tokens) if token == "[ent]"
            ]

            prediction_indicies = torch.tensor(predicted_indices)
            answer_onehot = torch.zeros_like(prediction_indicies, dtype=torch.float)
            answer_onehot[answer_index] = 1.0

            return {
                "candidate_ids": torch.tensor(candidate_ids).unsqueeze(0),
                "support_ids": torch.tensor(support_ids).unsqueeze(0),
                "prediction_indicies": prediction_indicies.unsqueeze(0),
                "correct_prediction_idx": answer_onehot.unsqueeze(0),
            }
        
        candidate_ids = []
        support_ids = []
        prediction_indicies = []
        correct_prediction_idx = []
        for v in instance:
            values = one_entry(v)
            candidate_ids.append(values["candidate_ids"])
            support_ids.append(values["support_ids"])
            prediction_indicies.append(values["prediction_indicies"])
            correct_prediction_idx.append(values["correct_prediction_idx"])
        
        return {
            "candidate_ids": torch.cat(candidate_ids, dim=0),
            "support_ids": torch.cat(support_ids, dim=0),
            "prediction_indicies": torch.cat(prediction_indicies, dim=0),
            "correct_prediction_idx": torch.cat(correct_prediction_idx, dim=0),
        }

class WikiHopDataset:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data_args,
        training_args,
        rename_to_text=False,
        use_wandb=True,
    ) -> None:
        super().__init__()
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb

        data_name = data_args.dataset_name
        assert data_name == "wikihop"
        try:
            path = "./data/qangaroo_v1.1/wikihop/{}.json"
            train = load_dataset("json", data_files=path.format("train"))["train"]
            validation = load_dataset("json", data_files=path.format("dev"))["train"]
            validation.remove_columns("annotations")
        except Exception as e:
            print("Error loading dataset. Extract the qangaroo_v1.1.zip file in the data folder, as described in the README.")
            raise e

        raw_datasets = DatasetDict({"train": train, "validation": validation})

        self.tokenizer = tokenizer
        self.data_args = data_args

        # labels
        print(raw_datasets)

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = False

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Processing Data",
            remove_columns=raw_datasets.column_names["train"],
            num_proc=8
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

        self.train_data_collator = DataCollatorForWikiHop(self.tokenizer, shuffle_candidates=True)
        self.eval_data_collator = DataCollatorForWikiHop(self.tokenizer, shuffle_candidates=False)

    def tok(self, s):
        return self.tokenizer.tokenize(normalize_string(s))

    def preprocess_function(self, example):
        doc_start = "</s>"
        doc_end = "</s>"

        query_tokens = ["[question]"] + self.tok(example["query"]) + ["[/question]"]
        supports_tokens = [
            [doc_start] + self.tok(support) + [doc_end]
            for support in example["supports"]
        ]
        candidate_tokens = [
            ["[ent]"] + self.tok(candidate) + ["[/ent]"]
            for candidate in example["candidates"]
        ]
        answer_index = example["candidates"].index(example["answer"])

        example["query_tokens"] = query_tokens
        example["supports_tokens"] = supports_tokens
        example["candidate_tokens"] = candidate_tokens
        example["answer_index"] = answer_index

        return example

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        real_values = p.label_ids.argmax(axis=1)
        result = {"accuracy": (preds == real_values).astype(np.float32).mean().item()}

        return result
