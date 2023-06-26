from transformers import LongformerPreTrainedModel, LongformerModel, RobertaPreTrainedModel, RobertaModel
from .longformer import LongformerWikiHopOutput
import torch
from typing import Optional, Union, Tuple
from torch import nn
from torch.nn import CrossEntropyLoss

class LongformerForWikiHop(LongformerPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"classifier", r"prefix_encoder", r"propagated_prefix_encoder"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.answer_score = nn.Linear(config.hidden_size, 1, bias=False)

        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        candidate_ids: Optional[torch.Tensor] = None,
        support_ids: Optional[torch.Tensor] = None,
        prediction_indicies: Optional[torch.Tensor] = None,
        correct_prediction_idx: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerWikiHopOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        candidate_len = candidate_ids.shape[1]
        support_len = support_ids.shape[1]
        max_seq_len = 4096

        if candidate_len + support_len <= max_seq_len:
            # regular
            input_ids = torch.cat([candidate_ids, support_ids], dim=1)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
            global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
            # globally attend to question tokens and candidate answers
            global_attention_mask[0, :candidate_len] = 1
            outputs = self.longformer(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                token_type_ids=None,
                head_mask=None,
                position_ids=None,
                inputs_embeds=None,
                return_dict=return_dict,
            )

            sequence_outputs = [outputs[0]]
        else:
            # batch together
            sequence_outputs = []
            available_support_len = max_seq_len - candidate_len
            for start in range(0, support_len, available_support_len):
                end = min(start + available_support_len, support_len)
                input_ids = torch.cat([candidate_ids, support_ids[:, start:end]], dim=1)
                attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
                global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

                # globally attend to question tokens and candidate answers
                global_attention_mask[0, :candidate_len] = 1

                outputs = self.longformer(
                    input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    token_type_ids=None,
                    head_mask=None,
                    position_ids=None,
                    inputs_embeds=None,
                    return_dict=return_dict,
                )
                sequence_outputs.append(outputs[0])

        prediction_activations = [act.index_select(1, prediction_indicies[0]) for act in sequence_outputs]
        prediction_scores = [
            self.answer_score(prediction_act).squeeze(-1)
            for prediction_act in prediction_activations
        ]
        # prediction_scores is a list of tensors, each is (batch_size, num_predictions)
        # sum across the list for each possible prediction
        sum_prediction_scores = torch.cat(
                [pred_scores.unsqueeze(-1) for pred_scores in prediction_scores], dim=-1
        ).sum(dim=-1)


        loss = None
        if correct_prediction_idx is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(sum_prediction_scores, correct_prediction_idx)

        if not return_dict:
            output = (sum_prediction_scores, sum_prediction_scores.argmax(dim=1))
            return ((loss,) + output) if loss is not None else output

        return LongformerWikiHopOutput(
            loss=loss,
            logits=sum_prediction_scores,
            predicted_answers=sum_prediction_scores.argmax(dim=1)
        )

class RobertaForWikiHop(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"classifier", r"prefix_encoder", r"propagated_prefix_encoder"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.answer_score = nn.Linear(config.hidden_size, 1, bias=False)

        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        candidate_ids: Optional[torch.Tensor] = None,
        support_ids: Optional[torch.Tensor] = None,
        prediction_indicies: Optional[torch.Tensor] = None,
        correct_prediction_idx: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerWikiHopOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        candidate_len = candidate_ids.shape[1]
        support_len = support_ids.shape[1]
        max_seq_len = 512

        if candidate_len + support_len <= max_seq_len:
            # regular
            input_ids = torch.cat([candidate_ids, support_ids], dim=1)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                head_mask=None,
                position_ids=None,
                inputs_embeds=None,
                return_dict=return_dict,
            )

            sequence_outputs = [outputs[0]]
        else:
            # batch together
            sequence_outputs = []
            available_support_len = max_seq_len - candidate_len
            for start in range(0, support_len, available_support_len):
                end = min(start + available_support_len, support_len)
                input_ids = torch.cat([candidate_ids, support_ids[:, start:end]], dim=1)
                attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

                outputs = self.roberta(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    head_mask=None,
                    position_ids=None,
                    inputs_embeds=None,
                    return_dict=return_dict,
                )
                sequence_outputs.append(outputs[0])

        prediction_activations = [act.index_select(1, prediction_indicies[0]) for act in sequence_outputs]
        prediction_scores = [
            self.answer_score(prediction_act).squeeze(-1)
            for prediction_act in prediction_activations
        ]
        # prediction_scores is a list of tensors, each is (batch_size, num_predictions)
        # sum across the list for each possible prediction
        sum_prediction_scores = torch.cat(
                [pred_scores.unsqueeze(-1) for pred_scores in prediction_scores], dim=-1
        ).sum(dim=-1)


        loss = None
        if correct_prediction_idx is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(sum_prediction_scores, correct_prediction_idx)

        if not return_dict:
            output = (sum_prediction_scores, sum_prediction_scores.argmax(dim=1))
            return ((loss,) + output) if loss is not None else output

        return LongformerWikiHopOutput(
            loss=loss,
            logits=sum_prediction_scores,
            predicted_answers=sum_prediction_scores.argmax(dim=1)
        )
