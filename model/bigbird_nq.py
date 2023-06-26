
from torch import nn
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import BigBirdPreTrainedModel, BigBirdModel
from .bigbird import BigBirdForQuestionAnsweringHead, BigBirdForNaturalQuestionsModelOutput


class BigBirdForNaturalQuestions(BigBirdPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels
        self.sep_token_id = config.sep_token_id

        self.bert = BigBirdModel(config, add_pooling_layer=add_pooling_layer)
        self.qa_classifier = BigBirdForQuestionAnsweringHead(config)
        self.answer_type_cls = nn.Linear(config.hidden_size, 5)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        question_lengths=None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        answer_type: Optional[torch.LongTensor] = None
    ) -> Union[BigBirdForNaturalQuestionsModelOutput, Tuple[torch.FloatTensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seqlen = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)

        if question_lengths is None and input_ids is not None:
            # assuming input_ids format: <cls> <question> <sep> context <sep>
            question_lengths = torch.argmax(input_ids.eq(self.sep_token_id).int(), dim=-1) + 1
            question_lengths.unsqueeze_(1)

        logits_mask = None
        if question_lengths is not None:
            # setting lengths logits to `-inf`
            logits_mask = self.prepare_question_mask(question_lengths, seqlen)
            if token_type_ids is None:
                token_type_ids = torch.ones(logits_mask.size(), dtype=int, device=logits_mask.device) - logits_mask
            logits_mask = logits_mask
            logits_mask[:, 0] = False
            logits_mask.unsqueeze_(2)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.qa_classifier(sequence_output)

        if logits_mask is not None:
            # removing question tokens from the competition
            logits = logits - logits_mask * 1e6

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        answer_type_logits = self.answer_type_cls(outputs.pooler_output)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            if answer_type is not None:
                cls_loss = loss_fct(answer_type_logits, answer_type)
                total_loss = (start_loss + end_loss + cls_loss) / 3
            else:
                total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BigBirdForNaturalQuestionsModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            answer_type=answer_type_logits
        )

    @staticmethod
    def prepare_question_mask(q_lengths: torch.Tensor, maxlen: int):
        # q_lengths -> (bz, 1)
        mask = torch.arange(0, maxlen).to(q_lengths.device)
        mask.unsqueeze_(0)  # -> (1, maxlen)
        mask = torch.where(mask < q_lengths, 1, 0)
        return mask
