from typing import List, Optional, Tuple, Union
import math

import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaSelfOutput,
    RobertaIntermediate,
    RobertaOutput,
    RobertaPooler,
)

from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers import RobertaPreTrainedModel 
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.utils import logging

from model.prefix_encoder import PrefixEncoder
from model.kernel_utils import get_kernel
from model.longformer import (
    LongformerClassificationHead,
    LongformerSequenceClassifierOutput,
    LongformerBaseModelOutputWithPooling,
    LongformerPreTrainedModel,
    LongformerPooler,
    LongformerIntermediate,
    LongformerOutput,
    LongformerSelfOutput,
    LongformerBaseModelOutput,
)

logger = logging.get_logger(__name__)


class RobertaKernelPrefixForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaKernelModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        for name, param in self.roberta.named_parameters():  
            if "scale_factor" in name:
                continue
            if "offset" in name:
                continue
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.add_pre_seq_len = config.add_pre_seq_len

        self.total_pre_len = self.pre_seq_len + (
            self.add_pre_seq_len if self.add_pre_seq_len else 0
        )

        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        if self.add_pre_seq_len:
            self.add_prefix_tokens = torch.arange(
                self.pre_seq_len, self.pre_seq_len + self.add_pre_seq_len
            ).long()

        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for name, param in self.roberta.named_parameters():
            if not param.requires_grad:
                bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print("total param is {}".format(total_param))
        print("takes {} of model".format(total_param / bert_param))

    def get_prompt(self, batch_size):
        prefix_tokens = (
            self.prefix_tokens.unsqueeze(0)
            .expand(batch_size, -1)
            .to(self.roberta.device)
        )

        if self.add_pre_seq_len:
            add_prefix_tokens = self.add_prefix_tokens
            past_key_values = self.prefix_encoder(prefix_tokens, add_prefix_tokens)
        else:
            past_key_values = self.prefix_encoder(prefix_tokens)

        past_key_values = past_key_values.view(
            batch_size, self.total_pre_len, self.n_layer * 2, self.n_head, self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.total_pre_len).to(
            self.roberta.device
        )
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaKernelModel(RobertaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaKernelEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class RobertaKernelEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaKernelLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class RobertaKernelLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaKernelAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RobertaKernelAttention(config, position_embedding_type="absolute")
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class RobertaKernelAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = RobertaKernelSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class RobertaKernelSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.kernel_scale = config.kernel_scale
        self.kernel_scale_init = config.kernel_scale_init

        # use sigmoid for alpha to be between [0, 1] when combining kernels
        # if self.kernel_scale is not None:
        #     self.scale_activation = nn.Sigmoid()

        if self.kernel_scale == "Scalar":
            self.scale_factor = nn.Parameter(
                    torch.tensor(float(self.kernel_scale_init)))
        elif self.kernel_scale == "Vector":
            self.scale_factor = nn.Parameter(
                    torch.as_tensor(np.full(64,
                        float(self.kernel_scale_init))).type(torch.float32))

        self.prefix_kernel_type = config.prefix_kernel
        if self.prefix_kernel_type != "Exp":
            self.prefix_kernel = get_kernel(self.prefix_kernel_type,
                    self.attention_head_size)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            prefix_key_layer = past_key_value[0]
            prefix_value_layer = past_key_value[1]
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        # attention of frozen model
        orig_attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # prefix attention
        # shape (bs, n_heads, seq_len, prefix_len)

        # Use kernel for most kernels but just matmul and softmax for exp
        if self.prefix_kernel_type != "Exp":
            kernel_results = self.prefix_kernel(query_layer,
                    prefix_key_layer).to_dense()
        else:
            kernel_results = torch.matmul(query_layer, 
                    prefix_key_layer.transpose(-1, -2))
        


        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key


        # Scale already accounted for by kernel lengthscale
        if self.prefix_kernel_type != "Exp":
            scaled_kernel_scores = kernel_results
        else:
            scaled_kernel_scores = kernel_results / math.sqrt(self.attention_head_size)

        orig_attention_scores = orig_attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            orig_attention_scores = orig_attention_scores + \
                    attention_mask[:, :, :, scaled_kernel_scores.shape[3]:]  # add attention mask for original sequence
            scaled_kernel_scores = scaled_kernel_scores + \
                    attention_mask[:, :, :, :scaled_kernel_scores.shape[3]]  # add first n tokens of attention mask to first n prefix tokens

        # normalize by dividing by sum to shift between [0, 1] for kernel
        if self.prefix_kernel_type != "Exp":
            norm = scaled_kernel_scores.sum(dim=3)
            normalized_kernel_scores = scaled_kernel_scores / norm.unsqueeze(3)
        else:
            normalized_kernel_scores = nn.functional.softmax(scaled_kernel_scores, dim=-1)

        # apply scale_factor
        normalized_kernel_scores = normalized_kernel_scores * self.scale_factor

        # Normalize the attention scores to probabilities.
        orig_attention_probs = nn.functional.softmax(orig_attention_scores, dim=-1)

        # No more since we are not using softmax kernel
        # prefix_attention_probs = nn.functional.softmax(normalized_kernel_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        orig_attention_probs = self.dropout(orig_attention_probs)
        normalized_kernel_scores = self.dropout(normalized_kernel_scores)

        # Mask heads if we want to
        if head_mask is not None:
            prefix_attention_probs = prefix_attention_probs * head_mask
            orig_attention_probs = orig_attention_probs * head_mask


        orig_context_layer = torch.matmul(orig_attention_probs, value_layer)
        prefix_context_layer = torch.matmul(normalized_kernel_scores, prefix_value_layer)

        # if self.kernel_scale is not None:
        #     # restrict depending on activation function
        #     # activated_scale = self.scale_activation(self.scale_factor)
        #     context_layer = orig_context_layer + \
        #                     (self.scale_factor * prefix_context_layer)
        # else:
        context_layer = orig_context_layer + prefix_context_layer

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class LongformerKernelSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        # separate projection layers for tokens with global attention
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.kernel_scale = config.kernel_scale
        self.kernel_scale_init = config.kernel_scale_init

        self.prefix_kernel_type = config.prefix_kernel
        if self.prefix_kernel_type != "Exp":
            self.prefix_kernel = get_kernel(self.prefix_kernel_type,
                    self.head_dim)


        if self.kernel_scale == "Scalar":
            # initialize to favour prefixes (still favours frozen overall since
            # seq_len >> pre_len
            self.scale_factor = nn.Parameter(
                    torch.tensor(float(self.kernel_scale_init)))
        elif self.kernel_scale == "Vector":
            self.scale_factor = nn.Parameter(
                    torch.as_tensor(np.full(64,
                        float(self.kernel_scale_init))).type(torch.float32))

        self.propagate_prefix = config.propagate_prefix
        self.pre_seq_len = config.pre_seq_len

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
        past_key_value=None,
    ):
        """
        [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.
        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:
            - -10000: no attention
            - 0: local attention
            - +10000: global attention
        """
        hidden_states = hidden_states.transpose(0, 1)

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        past_key = None
        past_value = None

        # Do not support both propagate and prefix-tuning 
        if past_key_value is not None and self.propagate_prefix == "none":
            past_key = past_key_value[0].permute(0, 2, 1, 3)
            past_value = past_key_value[1].permute(0, 2, 1, 3)


        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim
        ).transpose(0, 1)
        key_vectors = key_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim
        ).transpose(0, 1)
        value_vectors = value_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim
        ).transpose(0, 1)


        # Split key and value if propagation
        if self.propagate_prefix != "none":
            key_split = torch.split(key_vectors, [self.pre_seq_len,
                key_vectors.shape[1]-self.pre_seq_len], dim=1)
            past_key = key_split[0]
            key_vectors = key_split[1]

            value_split = torch.split(value_vectors, [self.pre_seq_len,
                value_vectors.shape[1]-self.pre_seq_len], dim=1)
            past_value = value_split[0]
            value_vectors = value_split[1]


        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        past_key_value_size = (
            past_key_value.shape[3] if past_key_value is not None else 0
        )
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

        attention_window = self.one_sided_attn_window_size * 2
        padding_len = (attention_window - seq_len % attention_window) % attention_window

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(
            query_vectors
        ).masked_fill(
            remove_from_windowed_attention_mask, torch.finfo(query_vectors.dtype).min
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()),
            float_mask,
            self.one_sided_attn_window_size,
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], (
            f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads},"
            f" {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"
        )

        # compute local attention probs from global attention keys and concat over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(
                is_index_global_attn,
            )

            # calculate global attn probs from global key
            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)

            # free memory
            del global_key_attn_scores
        
        if self.prefix_kernel_type != "Exp":
            scaled_kernel_scores = self.prefix_kernel(query_vectors.transpose(1,2),
                    past_key.transpose(1,2)).to_dense() # ()
            norm = scaled_kernel_scores.sum(dim=3)
            normalized_kernel_scores = scaled_kernel_scores / norm.unsqueeze(3)
        else:
            kernel_results = torch.matmul(query_vectors.transpose(1,2),
                    past_key.transpose(1,2).transpose(-1,-2))
            scaled_kernel_scores = kernel_results / math.sqrt(self.head_dim)
            normalized_kernel_scores = nn.functional.softmax(scaled_kernel_scores, dim=-1)

        normalized_kernel_scores = normalized_kernel_scores * self.scale_factor

        attn_probs = nn.functional.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(
            attn_probs, is_index_masked[:, :, None, None], 0.0
        )
        attn_probs = attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores

        # apply dropout
        attn_probs = nn.functional.dropout(
            attn_probs, p=self.dropout, training=self.training
        )
        prefix_attn_probs = nn.functional.dropout(
            normalized_kernel_scores, p=self.dropout, training=self.training
        )

        prefix_attn_probs = torch.matmul(normalized_kernel_scores, past_value.transpose(1,2))
        prefix_attn_probs = prefix_attn_probs.permute((2, 0, 1, 3))
        sl, bs, heads, head_dim = prefix_attn_probs.shape
        prefix_attn_probs = prefix_attn_probs.reshape((sl, bs, heads*head_dim))

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ), "Unexpected size"
        attn_output = (
            attn_output.transpose(0, 1)
            .reshape(seq_len, batch_size, embed_dim)
            .contiguous()
        )

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            (
                global_attn_output,
                global_attn_probs,
            ) = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0],
                :,
                is_local_index_global_attn_nonzero[1],
            ]

            # overwrite values with global attention
            attn_output[
                is_index_global_attn_nonzero[::-1]
            ] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0

        attn_output = attn_output + prefix_attn_probs

        outputs = (attn_output.transpose(0, 1),)

        if output_attentions:
            outputs += (attn_probs,)

        return (
            outputs + (global_attn_probs,)
            if (is_global_attn and output_attentions)
            else outputs
        )

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = nn.functional.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2],
            hidden_states_padded.size(-1),
            hidden_states_padded.size(-2),
        )
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example:
        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```
                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        (
            total_num_heads,
            num_chunks,
            window_overlap,
            hidden_dim,
        ) = chunked_hidden_states.size()
        chunked_hidden_states = nn.functional.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""

        # non-overlapping chunks of size = 2w
        hidden_states = hidden_states.view(
            hidden_states.size(0),
            hidden_states.size(1) // (window_overlap * 2),
            window_overlap * 2,
            hidden_states.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(hidden_states.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(hidden_states.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = (
            input_tensor.new_ones(affected_seq_len, affected_seq_len + 1)
            .tril()
            .flip(dims=[0])
        )
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        beginning_input.masked_fill_(
            beginning_mask == 1, -float("inf")
        )  # `== 1` converts to bool or uint8
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(
            ending_mask == 1, -float("inf")
        )  # `== 1` converts to bool or uint8

    def _sliding_chunks_query_key_matmul(
        self, query: torch.Tensor, key: torch.Tensor, window_overlap: int
    ):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
            seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        query = self._chunk(query, window_overlap)
        key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (query, key)
        )  # multiply

        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (
                batch_size * num_heads,
                chunks_count + 1,
                window_overlap,
                window_overlap * 2 + 1,
            )
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[
            :, :-1, :, window_overlap:
        ] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[
            :, -1, :, window_overlap:
        ] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[
            :, 1:, :, :window_overlap
        ] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[
            :, 0, 1:window_overlap, 1:window_overlap
        ] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        batch_size, seq_len, num_heads, head_dim = value.size()

        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads,
            seq_len // window_overlap,
            window_overlap,
            2 * window_overlap + 1,
        )

        # group batch_size and num_heads dimensions into one
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(
            value, (0, 0, window_overlap, window_overlap), value=-1
        )

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (
            batch_size * num_heads,
            chunks_count + 1,
            3 * window_overlap,
            head_dim,
        )
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(
            size=chunked_value_size, stride=chunked_value_stride
        )

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

    @staticmethod
    def _get_global_attn_indices_custom(is_index_global_attn, prefix_length=None):
        """compute global attn indices required throughout forward pass"""
        batch_size = is_index_global_attn.shape[0]
        prefix_length = (prefix_length if prefix_length is not None else 0)

        # helper variable
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1) + prefix_length

        # max number of global attn indices in batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # helper variable
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # indices of global attn
        is_index_global_attn_nonzero = is_index_global_attn.nonzero()
        v = torch.arange(0, prefix_length, device=is_index_global_attn.device).repeat((batch_size, 1))
        p = torch.arange(0, batch_size, device=is_index_global_attn.device).repeat((prefix_length, 1)).T
        t = torch.full((batch_size, prefix_length), is_index_global_attn.shape[1])
        additional_values = torch.stack((p,v+t), dim=2).reshape((-1, 2))
        is_index_global_attn_nonzero = tuple(t.squeeze() for t in torch.cat((is_index_global_attn_nonzero, additional_values), dim=0).split(1, dim=1))

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero()
        is_local_index_global_attn_nonzero = tuple(t.squeeze() for t in is_local_index_global_attn_nonzero.split(1, dim=1))

        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (
            is_local_index_global_attn == 0
        ).nonzero(as_tuple=True)

        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # helper variable
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # max number of global attn indices in batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # indices of global attn
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

        # helper variable
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
        self,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]

        # create only global key vectors
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )

        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[
            is_index_global_attn_nonzero
        ]

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum(
            "blhd,bshd->blhs", (query_vectors, key_vectors_only_global)
        )

        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0],
            :,
            :,
            is_local_index_no_global_attn_nonzero[1],
        ] = torch.finfo(attn_probs_from_global_key.dtype).min

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
        batch_size = attn_probs.shape[0]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        # get value vectors for global only
        value_vectors_only_global = value_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[
            is_index_global_attn_nonzero
        ]

        # use `matmul` because `einsum` crashes sometimes with fp16
        # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
        # compute attn output only global
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.transpose(1, 2).clone(),
            value_vectors_only_global.transpose(1, 2).clone(),
        ).transpose(1, 2)

        # reshape attn probs
        attn_probs_without_global = attn_probs.narrow(
            -1,
            max_num_global_attn_indices,
            attn_probs.size(-1) - max_num_global_attn_indices,
        ).contiguous()

        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors[:, :, :, :], self.one_sided_attn_window_size
        )
        return attn_output_only_global + attn_output_without_global

    def _transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + ()
        x = x.view(*new_x_shape)
        return x.permute(1, 2, 0, 3)

    def _compute_global_attn_output_from_hidden(
        self,
        hidden_states,
        max_num_global_attn_indices,
        layer_head_mask,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
    ):
        # TODO: add prefix tuning to this function
        seq_len, batch_size = hidden_states.shape[:2]

        # prepare global hidden states
        global_attn_hidden_states = hidden_states.new_zeros(
            max_num_global_attn_indices, batch_size, self.embed_dim
        )
        global_attn_hidden_states[
            is_local_index_global_attn_nonzero[::-1]
        ] = hidden_states[is_index_global_attn_nonzero[::-1]]

        # global key, query, value
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)
        head_shape = self.num_heads * self.head_dim
        
        # normalize
        global_query_vectors_only_global /= math.sqrt(self.head_dim)

        # reshape
        global_query_vectors_only_global = (
            global_query_vectors_only_global.contiguous()
            .view(
                max_num_global_attn_indices, batch_size * self.num_heads, self.head_dim
            )
            .transpose(0, 1)
        )  # (batch_size * self.num_heads, max_num_global_attn_indices, head_dim)
        global_key_vectors = (
            global_key_vectors.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )  # batch_size * self.num_heads, seq_len, head_dim)
        global_value_vectors = (
            global_value_vectors.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )  # batch_size * self.num_heads, seq_len, head_dim)

        # compute attn scores
        global_attn_scores = torch.bmm(
            global_query_vectors_only_global, global_key_vectors.transpose(1, 2)
        )

        assert list(global_attn_scores.size()) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            seq_len,
        ], (
            "global_attn_scores have the wrong size. Size should be"
            f" {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is"
            f" {global_attn_scores.size()}."
        )

        global_attn_scores = global_attn_scores.view(
            batch_size, self.num_heads, max_num_global_attn_indices, seq_len
        )

        global_attn_scores[
            is_local_index_no_global_attn_nonzero[0],
            :,
            is_local_index_no_global_attn_nonzero[1],
            :,
        ] = torch.finfo(global_attn_scores.dtype).min

        global_attn_scores = global_attn_scores.masked_fill(
            is_index_masked[:, None, None, :],
            torch.finfo(global_attn_scores.dtype).min,
        )

        global_attn_scores = global_attn_scores.view(
            batch_size * self.num_heads, max_num_global_attn_indices, seq_len
        )

        # compute global attn probs
        global_attn_probs_float = nn.functional.softmax(
            global_attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        # apply layer head masking
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            global_attn_probs_float = layer_head_mask.view(
                1, -1, 1, 1
            ) * global_attn_probs_float.view(
                batch_size, self.num_heads, max_num_global_attn_indices, seq_len
            )
            global_attn_probs_float = global_attn_probs_float.view(
                batch_size * self.num_heads, max_num_global_attn_indices, seq_len
            )

        global_attn_probs = nn.functional.dropout(
            global_attn_probs_float.type_as(global_attn_scores),
            p=self.dropout,
            training=self.training,
        )

        # global attn output
        global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)

        assert list(global_attn_output.size()) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            self.head_dim,
        ], (
            "global_attn_output tensor has the wrong size. Size should be"
            f" {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is"
            f" {global_attn_output.size()}."
        )

        global_attn_probs = global_attn_probs.view(
            batch_size, self.num_heads, max_num_global_attn_indices, seq_len
        )
        global_attn_output = global_attn_output.view(
            batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim
        )
        return global_attn_output, global_attn_probs


class LongformerKernelAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.self = LongformerKernelSelfAttention(config, layer_id)
        self.output = LongformerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
        past_key_value=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs


class LongformerKernelEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                LongformerKernelLayer(config, layer_id=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        padding_len=0,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        past_key_values=None,
        additional_prefixes=None
    ):

        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None  # All local attentions.
        all_global_attentions = () if (output_attentions and is_global_attn) else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layer)
            ), f"The head_mask should be specified for {len(self.layer)} layers, but it is for {head_mask.size()[0]}."

        for idx, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            additional_prefix = additional_prefixes[idx] if additional_prefixes is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, is_global_attn, output_attentions, past_key_value, additional_prefix)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    is_index_masked,
                    is_index_global_attn,
                    use_reentrant=False
                )

            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=head_mask[idx] if head_mask is not None else None,
                    is_index_masked=is_index_masked,
                    is_index_global_attn=is_index_global_attn,
                    is_global_attn=is_global_attn,
                    output_attentions=output_attentions,
                    past_key_value=past_key_value,
                    propagated_prefix=additional_prefix
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len + 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)

                if is_global_attn:
                    # bzs x num_attn_heads x num_global_attn x seq_len => bzs x num_attn_heads x seq_len x num_global_attn
                    all_global_attentions = all_global_attentions + (
                        layer_outputs[2].transpose(2, 3),
                    )

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # undo padding
        if padding_len > 0:
            # unpad `hidden_states` because the calling function is expecting a length == input_ids.size(1)
            hidden_states = hidden_states[:, :-padding_len]
            if output_hidden_states:
                all_hidden_states = tuple(
                    [state[:, :-padding_len] for state in all_hidden_states]
                )

            if output_attentions:
                all_attentions = tuple(
                    [state[:, :, :-padding_len, :] for state in all_attentions]
                )

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_attentions,
                    all_global_attentions,
                ]
                if v is not None
            )
        return LongformerBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            global_attentions=all_global_attentions,
        )

class LongformerKernelModel(LongformerPreTrainedModel):
    """
    This class copied code from [`RobertaModel`] and overwrote standard self-attention with longformer self-attention
    to provide the ability to process long sequences following the self-attention approach described in [Longformer:
    the Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, and Arman Cohan.
    Longformer self-attention combines a local (sliding window) and global attention to extend to long documents
    without the O(n^2) increase in memory and compute.
    The self-attention module `LongformerSelfAttention` implemented here supports the combination of local and global
    attention but it lacks support for autoregressive attention and dilated attention. Autoregressive and dilated
    attention are more relevant for autoregressive language modeling than finetuning on downstream tasks. Future
    release will add support for autoregressive attention, but the support for dilated attention requires a custom CUDA
    kernel to be memory and compute efficient.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        if isinstance(config.attention_window, int):
            assert (
                config.attention_window % 2 == 0
            ), "`config.attention_window` has to be an even value"
            assert (
                config.attention_window > 0
            ), "`config.attention_window` has to be positive"
            config.attention_window = [
                config.attention_window
            ] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.embeddings = LongformerEmbeddings(config)
        self.encoder = LongformerKernelEncoder(config)
        self.pooler = LongformerPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
        propagated_prefix_len: Optional[int]
    ):
        # XXX: important!
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert (
            attention_window % 2 == 0
        ), f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        if propagated_prefix_len is not None:
            seq_len += propagated_prefix_len

        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logger.info(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if input_ids is not None:
                input_ids = nn.functional.pad(
                    input_ids, (0, padding_len), value=pad_token_id
                )
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = nn.functional.pad(
                    position_ids, (0, padding_len), value=pad_token_id
                )
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat(
                    [inputs_embeds, inputs_embeds_padding], dim=-2
                )

            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens
            token_type_ids = nn.functional.pad(
                token_type_ids, (0, padding_len), value=0
            )  # pad with token_type_id = 0

        return (
            padding_len,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds,
        )

    def _merge_to_attention_mask(
        self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor
    ):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        propagated_prefix: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, LongformerBaseModelOutputWithPooling]:
        r"""
        Returns:
        Examples:
        ```python
        >>> import torch
        >>> from transformers import LongformerModel, LongformerTokenizer
        >>> model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        >>> tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        >>> SAMPLE_TEXT = " ".join(["Hello world! "] * 1000)  # long input document
        >>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1
        >>> attention_mask = torch.ones(
        ...     input_ids.shape, dtype=torch.long, device=input_ids.device
        ... )  # initialize to local attention
        >>> global_attention_mask = torch.zeros(
        ...     input_ids.shape, dtype=torch.long, device=input_ids.device
        ... )  # initialize to global attention to be deactivated for all tokens
        >>> global_attention_mask[
        ...     :,
        ...     [
        ...         1,
        ...         4,
        ...         21,
        ...     ],
        ... ] = 1  # Set global attention to random tokens for the sake of this example
        >>> # Usually, set global attention based on the task. For example,
        >>> # classification: the <s> token
        >>> # QA: question tokens
        >>> # LM: potentially on the beginning of sentences and paragraphs
        >>> outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        >>> sequence_output = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output
        ```"""

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        elif propagated_prefix is not None:
            past_key_values_length = propagated_prefix[0].shape[2]

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), device=device
            )
        if token_type_ids is None:
            shape = (input_shape[0], input_shape[1]+(past_key_values_length if propagated_prefix is not None else 0))
            token_type_ids = torch.zeros(shape, dtype=torch.long, device=device)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(
                attention_mask, global_attention_mask
            )

        (
            padding_len,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds,
        ) = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
            propagated_prefix_len=propagated_prefix.shape[3] if propagated_prefix is not None else None
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )[:, 0, 0, :]

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            add_additional_prefix=propagated_prefix is not None
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            padding_len=padding_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
            additional_prefixes=propagated_prefix
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return LongformerBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            global_attentions=encoder_outputs.global_attentions,
        )


class LongformerKernelPrefixForSequenceClassification(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.longformer = LongformerKernelModel(config, add_pooling_layer=False)

        for name, param in self.longformer.named_parameters():
            if "token_type_embeddings" in name:
                # don't freeze token type embeddings
                continue
            param.requires_grad = False
        
        self.classifier = LongformerClassificationHead(config)

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        self.propagate_prefix = self.config.propagate_prefix

        self.prefix_encoder = None
        self.propagated_prefix_encoder = None

        if self.propagate_prefix in ["none", "combine"]:
            self.prefix_encoder = PrefixEncoder(config, propagate_prefix=False)
        if self.propagate_prefix in ["only", "combine"]:
            self.propagated_prefix_encoder = PrefixEncoder(config, propagate_prefix=True)

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # Initialize weights and apply final processing
        self.post_init()

    def get_prompt(self, batch_size):
        prefix_tokens = (
            self.prefix_tokens.unsqueeze(0)
            .expand(batch_size, -1)
            .to(self.longformer.device)
        )

        prefix_past_key_values = self.prefix_encoder(prefix_tokens) if self.prefix_encoder else None
        propagated_past_key_values = self.propagated_prefix_encoder(prefix_tokens) if self.propagated_prefix_encoder else None

        # Number of matricies to use (k and v if no propagation, embeddings only if propacation)
        if prefix_past_key_values is not None:
            # Prefix only
            prefix_past_key_values = prefix_past_key_values.view(
                batch_size, self.pre_seq_len, self.n_layer * 2, self.n_head, self.n_embd
            )
            prefix_past_key_values = self.dropout(prefix_past_key_values)
            # n_layer, batch_size, n_head, total_prefix_len, hidden_size
            prefix_past_key_values = prefix_past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        if propagated_past_key_values is not None:
            # Propagated prefix only
            propagated_past_key_values = propagated_past_key_values.view(
                batch_size, self.pre_seq_len, self.n_layer*(2 if self.config.propagate_prefix_scalar else 1), self.config.hidden_size
            )
            propagated_past_key_values = self.dropout(propagated_past_key_values)
            # n_layer, batch_size, total_prefix_len, hidden_size
            propagated_past_key_values = propagated_past_key_values.permute([2, 0, 1, 3])

            if self.config.propagate_prefix_scalar:
                propagated_past_key_values = propagated_past_key_values.split(2)
            else:
                propagated_past_key_values = propagated_past_key_values.unsqueeze(1)

        return (prefix_past_key_values, propagated_past_key_values)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if global_attention_mask is None:
            # logger.info("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        # XXX: https://github.com/huggingface/transformers/issues/16811

        batch_size = input_ids.shape[0]
        (prefix_past_key_values, propagated_past_key_values) = self.get_prompt(batch_size=batch_size)

        if propagated_past_key_values is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_ids.shape, dtype=torch.int32, device=self.longformer.device)
            token_type_ids = torch.cat((torch.zeros((batch_size, self.pre_seq_len), dtype=torch.int32, device=self.longformer.device), token_type_ids), dim=1)

            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len, device=self.longformer.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            global_attention_mask = torch.cat((prefix_attention_mask, global_attention_mask), dim=1)

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=prefix_past_key_values,
            propagated_prefix=propagated_past_key_values,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )


class LongformerEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.additional_word_embeddings = None
        if config.additional_non_frozen_embeds is not None:
            self.additional_word_embeddings = nn.Embedding(
                config.additional_non_frozen_embeds, config.hidden_size
            )

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
        add_additional_prefix=False
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
                if add_additional_prefix:
                    additional_position_ids = torch.arange(0, past_key_values_length, device=input_ids.device).unsqueeze(0).repeat(input_ids.shape[0], 1)
                    position_ids = torch.cat((additional_position_ids, position_ids), dim=1)
            else:
                if add_additional_prefix:
                    raise NotImplementedError("input_ids not passed unimplemented for propagation prefix tuning")
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds, past_key_values_length
                )

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            if add_additional_prefix:
                raise NotImplementedError("input_ids not passed unimplemented for propagation prefix tuning")
            input_shape = inputs_embeds.size()[:-1]

        # If we propagate the prefix, actually add an embedding to the start of the model
        seq_length = input_shape[1] + (past_key_values_length if add_additional_prefix is not None else 0)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (input_shape[0], seq_length), dtype=torch.long, device=position_ids.device
            )

        if inputs_embeds is None:
            if self.additional_word_embeddings is not None:
                mask = input_ids >= self.word_embeddings.num_embeddings
                masked = input_ids[mask].clone()
                input_ids[mask] = 0
                inputs_embeds = self.word_embeddings(input_ids)
                inputs_embeds[mask, :] = self.additional_word_embeddings(masked-self.word_embeddings.num_embeddings)
            else:
                inputs_embeds = self.word_embeddings(input_ids)
            if add_additional_prefix:
                # This prompt is zeros, as we add the prefix in the RobertaLayer.forward
                prompt_embeds = torch.zeros((inputs_embeds.shape[0], past_key_values_length, inputs_embeds.shape[2]), dtype=torch.int32, device=inputs_embeds.device)
                # Prepend our additional prompt
                inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(
        self, inputs_embeds, past_key_values_length: int
    ):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor inputs_embeds:
        Returns: torch.Tensor
        """
        raise NotImplementedError(
            "Position ids from input embeds is not implemented yet!"
        )
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class LongformerKernelLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.attention = LongformerKernelAttention(config, layer_id)
        self.intermediate = LongformerIntermediate(config)
        self.output = LongformerOutput(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
        past_key_value=None,
        propagated_prefix=None
    ):
        if propagated_prefix is not None:
            hidden_states[:, :propagated_prefix[0].shape[1], :] += propagated_prefix[0]

        self_attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
        )
        attn_output = self_attn_outputs[0]
        outputs = self_attn_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def ff_chunk(self, attn_output):
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        return layer_output


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (
        torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
    ) * mask
    return incremental_indices.long() + padding_idx

