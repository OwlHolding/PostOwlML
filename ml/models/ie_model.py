import torch
from torch import nn


class ItemEmbeddingModel(nn.Module):
    def __init__(self, encoder_size, embedding_size, intermediate_sizes, text_encoder=None, image_encoder=None, video_encoder=None, audio_encoder=None):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.video_encoder = video_encoder    # not implemented
        self.audio_encoder = audio_encoder    # not implemented
        self.encoder_size = encoder_size
        self.intermediate_sizes = intermediate_sizes
        self.embedding_size = embedding_size
        mlp_activations = [nn.LeakyReLU for _ in range(len(self.intermediate_sizes))]
        encoder_dim = self.intermediate_sizes + [self.embedding_size]
        self.mlp = nn.Sequential(*self._spec2seq(
            self.encoder_size,
            encoder_dim,
            mlp_activations))

    def forward(self, x, content_type='text'):
        if content_type == 'text':
            attention_mask = x['attention_mask']
            item_universal_embedding = self.text_encoder(**x).last_hidden_state
            item_universal_embedding = self.mean_pooling(item_universal_embedding, attention_mask)
            item_universal_embedding = self.mlp(item_universal_embedding)
        elif content_type == 'image':
            item_universal_embedding = self.image_encoder(**x).last_hidden_state
            item_universal_embedding = self.mean_pooling(
                item_universal_embedding,
                torch.ones(item_universal_embedding.size()[:-1])
            )
            item_universal_embedding = self.mlp(item_universal_embedding)
        elif content_type == 'audio':
            item_universal_embedding = self.audio_encoder(**x).last_hidden_state
            item_universal_embedding = self.mean_pooling(
                item_universal_embedding,
                torch.ones(item_universal_embedding.size()[:-1])
            )
            item_universal_embedding = self.mlp(item_universal_embedding)
        elif content_type == 'video':
            item_universal_embedding = self.video_encoder(**x).last_hidden_state
            item_universal_embedding = self.mean_pooling(
                item_universal_embedding,
                torch.ones(item_universal_embedding.size()[:-1])
            )
            item_universal_embedding = self.mlp(item_universal_embedding)
        else:
            raise ValueError
        return item_universal_embedding

    @staticmethod
    def mean_pooling(last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

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
