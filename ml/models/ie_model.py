import torch
from torch import nn


class ItemEmbeddingModel(nn.Module):
    def __init__(self, encoder_size, embedding_size, intermediate_sizes, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.encoder_size = encoder_size
        self.intermediate_sizes = intermediate_sizes
        self.embedding_size = embedding_size
        mlp_activations = [nn.LeakyReLU for _ in range(len(self.intermediate_sizes))]
        encoder_dim = self.intermediate_sizes + [self.embedding_size]
        self.mlp = nn.Sequential(*self._spec2seq(
            self.encoder_size,
            encoder_dim,
            mlp_activations))
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, embedding):
        item_universal_embedding = self.mlp(embedding)
        item_universal_embedding = self.dropout(item_universal_embedding)
        return item_universal_embedding


    @staticmethod
    def _spec2seq(input_, dimensions, activations):
        layers = []
        for dim, act in zip(dimensions, activations):
            layer = nn.Linear(input_, dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if act:
                layers.append(act())
            input_ = dim
        return layers