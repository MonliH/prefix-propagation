import torch


class PrefixEncoder(torch.nn.Module):
    r"""
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config, propagate_prefix: bool = False):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            if propagate_prefix:
                # Use a two-layer MLP to encode the prefix
                self.trans = torch.nn.Sequential(
                    torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(
                        config.prefix_hidden_size,
                        config.num_hidden_layers * config.hidden_size,
                    ),
                )
            else:
                # Use a two-layer MLP to encode the prefix
                self.trans = torch.nn.Sequential(
                    torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(
                        config.prefix_hidden_size,
                        config.num_hidden_layers * 2 * config.hidden_size,
                    ),
                )
        elif not propagate_prefix:
            self.embedding = torch.nn.Embedding(
                config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size
            )
            self.trainable_embedding = None
            if config.add_pre_seq_len:
                self.trainable_embedding = torch.nn.Embedding(
                    config.add_pre_seq_len,
                    config.num_hidden_layers * 2 * config.hidden_size,
                )
                self.embedding.requires_grad = False
        elif propagate_prefix:
            self.embedding = torch.nn.Embedding(
                config.pre_seq_len, config.num_hidden_layers * config.hidden_size * (2 if config.propagate_prefix_scalar else 1)
            )
            self.trainable_embedding = None

    def forward(self, prefix: torch.Tensor, extra_prefix: torch.Tensor = None):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
            if extra_prefix is not None:
                assert False, "this should not happen?"
                if not self.trainable_embedding:
                    raise ValueError(
                        "Must have trainable embedding set (extra_prefix is given, but add_pre_seq_len is None)"
                    )
                past_key_values_trainable = self.trainable_embedding(prefix)
                past_key_values = torch.cat(
                    (past_key_values, past_key_values_trainable), 1
                )
        return past_key_values
