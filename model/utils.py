from enum import Enum

from transformers import (AutoConfig, AutoModelForMaskedLM,
                          AutoModelForMultipleChoice,
                          AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification)

from model.bigbird import (BigBirdPrefixForNaturalQuestions,
                           BigBirdPrefixForQuestionAnswering,
                           BigBirdPrefixForSequenceClassification,
                           BigBirdPrefixForTriviaQA,
                           BigBirdPrefixForWikiHop)
from model.kernel_models import (
    RobertaKernelPrefixForSequenceClassification,
    LongformerKernelPrefixForSequenceClassification,
)

from model.language_modeling import (BertPrefixForMaskedLM,
                                     RobertaPrefixForMaskedLM)
from model.longformer import LongformerPrefixForSequenceClassification, LongformerPrefixForWikiHop
from model.multiple_choice import (BertPrefixForMultipleChoice,
                                   BertPromptForMultipleChoice,
                                   DebertaPrefixForMultipleChoice,
                                   RobertaPrefixForMultipleChoice,
                                   RobertaPromptForMultipleChoice)
from model.question_answering import (BertPrefixForQuestionAnswering,
                                      DebertaPrefixModelForQuestionAnswering,
                                      RobertaPrefixModelForQuestionAnswering)
from model.roberta import RobertaPrefixForSequenceClassification
from model.sequence_classification import (
    BertPrefixForSequenceClassification, BertPromptForSequenceClassification,
    DebertaPrefixForSequenceClassification,
    RobertaPromptForSequenceClassification)
from model.token_classification import (BertPrefixForTokenClassification,
                                        DebertaPrefixForTokenClassification,
                                        DebertaV2PrefixForTokenClassification,
                                        RobertaPrefixForTokenClassification)

import torch
from torch import nn


class TaskType(Enum):
    TOKEN_CLASSIFICATION = (1,)
    SEQUENCE_CLASSIFICATION = (2,)
    QUESTION_ANSWERING = (3,)
    MULTIPLE_CHOICE = 4
    MASKED_LM = 5
    NATURAL_QUESTIONS = 6
    TRIVIA_QA = 7
    WIKI_HOP = 8


PREFIX_MODELS = {
    "bert": {
        TaskType.TOKEN_CLASSIFICATION: BertPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: BertPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: BertPrefixForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: BertPrefixForMultipleChoice,
        TaskType.MASKED_LM: BertPrefixForMaskedLM,
    },
    "roberta": {
        TaskType.TOKEN_CLASSIFICATION: RobertaPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: RobertaPrefixModelForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: RobertaPrefixForMultipleChoice,
        TaskType.MASKED_LM: RobertaPrefixForMaskedLM,
    },
    "deberta": {
        TaskType.TOKEN_CLASSIFICATION: DebertaPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: DebertaPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: DebertaPrefixModelForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: DebertaPrefixForMultipleChoice,
    },
    "deberta-v2": {
        TaskType.TOKEN_CLASSIFICATION: DebertaV2PrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: None,
        TaskType.QUESTION_ANSWERING: None,
        TaskType.MULTIPLE_CHOICE: None,
    },
    "longformer": {
        TaskType.SEQUENCE_CLASSIFICATION: LongformerPrefixForSequenceClassification,
        TaskType.WIKI_HOP: LongformerPrefixForWikiHop
    },
    "big_bird": {
        TaskType.SEQUENCE_CLASSIFICATION: BigBirdPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: BigBirdPrefixForQuestionAnswering,
        TaskType.NATURAL_QUESTIONS: BigBirdPrefixForNaturalQuestions,
        TaskType.TRIVIA_QA: BigBirdPrefixForTriviaQA,
        TaskType.WIKI_HOP: BigBirdPrefixForWikiHop
    },
}

PROMPT_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: BertPromptForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: BertPromptForMultipleChoice,
    },
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPromptForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: RobertaPromptForMultipleChoice,
    },
}

KERNEL_MODELS = {
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaKernelPrefixForSequenceClassification,
    },
    "longformer": {
        TaskType.SEQUENCE_CLASSIFICATION: LongformerKernelPrefixForSequenceClassification,
    },
}

AUTO_MODELS = {
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
    TaskType.MASKED_LM: AutoModelForMaskedLM,
}


def get_model(
    model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False, tokenizer = None 
):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.add_pre_seq_len = model_args.add_pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size
        config.propagate_prefix = model_args.propagate_prefix
        config.propagate_prefix_scalar = model_args.propagate_prefix_scalar
        config.additional_non_frozen_embeds = model_args.additional_non_frozen_embeds
        config.use_offset_pool_tok = model_args.use_offset_pool_tok

        if model_args.kernel:
            config.prefix_kernel = model_args.prefix_kernel
            config.kernel_scale = model_args.kernel_scale
            config.kernel_scale_init = model_args.kernel_scale_init
            model_class = KERNEL_MODELS[config.model_type][task_type]
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )

        else:
            model_class = PREFIX_MODELS[config.model_type][task_type]
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
    elif model_args.prompt:
        config.pre_seq_len = model_args.pre_seq_len
        model_class = PROMPT_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    else:
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
        # print(model.config)
        # print(model_args)
        # print(model)

        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "deberta":
                for param in model.deberta.parameters():
                    param.requires_grad = False
                for _, param in model.deberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print("***** total param is {} *****".format(total_param))
    return model
