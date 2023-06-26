from enum import Enum
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Literal

from transformers import HfArgumentParser, TrainingArguments

from tasks.utils import *


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """

    task_name: str = field(
        metadata={
            "help": "The name of the task to train on: " + ", ".join(TASKS),
            "choices": TASKS,
        },
    )
    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset to use: " + ", ".join(DATASETS),
            "choices": DATASETS,
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )
    template_id: Optional[int] = field(
        default=0, metadata={"help": "The specific prompt string to use"}
    )
    mask_prob: Optional[float] = field(
        default=0.15,
        metadata={"help": "The masking probability to use for masked LM task."},
    )

    early_stopping_patience: Optional[int] = field(
        default=-1,
        metadata={
            "help": "If default or less than 0, no early stopping."
            "Metric to monitor defaults to first in eval dictionary"
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    prefix: bool = field(
        default=False, metadata={"help": "Will use P-tuning v2 during training"}
    )
    prompt: bool = field(
        default=False, metadata={"help": "Will use prompt tuning during training"}
    )
    finetune: bool = field(
        default=False, metadata={"help": "Will use regular finetune during training"}
    )
    propagate_prefix: str = field(
        default="none",
        metadata={
            "help": "Will propagate query of prefix (increases parameter count)."
            "Set to `only` for prefix propagation only, `none` to disable, "
            "or `combine` to combine it with prefix tuning (half of the prefix length will "
            "be used for actual prefix tuning, and half for propagated tokens)",
            "choices": ["none", "only", "combine"],
        },
    )
    use_offset_pool_tok: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the [cls] token for pooling or the first token "
            "(which is a prefix for prefix propagation). Doesn't change anthing for "
            "non-prefix propagation models"
        }
    )
    propagate_prefix_scalar: bool = field(
        default=False,
        metadata={"help": "Add a scaling term to the propagated prefixes"},
    )
    pre_seq_len: int = field(default=4, metadata={"help": "The length of prompt"})
    add_pre_seq_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "The length of added prompt. If set to a value other than None, freeze other prompts"
        },
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Apply a two-layer MLP head over the prefix embeddings"},
    )
    prefix_hidden_size: int = field(
        default=512,
        metadata={
            "help": "The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used"
        },
    )
    hidden_dropout_prob: float = field(
        default=0.1, metadata={"help": "The dropout probability used in the models"}
    )

    kernel: bool = field(
        default=False,
        metadata={
            "help": "Use kernel composition of prefix\
                with original sequence"
        },
    )
    prefix_kernel: Optional[str] = field(
        default="RBF",
        metadata={"help": "Prefix kernel type",
                  "choices": ['RBF','Poly', 'Exp']},
    )
    kernel_scale: Optional[str] = field(
        default=None,
        metadata={"help": "Type of scale factor",
                  "choices": [None, 'Scalar', 'Vector']},
    )
    kernel_scale_init: Optional[float] = field(
        default=5.0,
        metadata={"help": "Initial scale factor value (same for each layer),"
            "must be float"},
    )

    additional_non_frozen_embeds: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Length of additional embeddings that should not be frozen."
                "Useful for tasks like wikihop which have extra special tokens that need to be trained"
            )
        },
    )

@dataclass
class QuestionAnwseringArguments:
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "The total number of n-best predictions to generate when looking for an answer."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    version_2_with_negative: bool = field(
        default=False,
        metadata={"help": "If true, some of the examples do not have an answer."},
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    do_hyper_search: bool = field(
        default=False, metadata={"help": "Run a hyperparameter search"}
    )


def get_args():
    """Parse all the args."""
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            CustomTrainingArguments,
            QuestionAnwseringArguments,
        )
    )

    args = parser.parse_args_into_dataclasses()

    return args
