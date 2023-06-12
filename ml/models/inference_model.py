import torch
from torch import nn

from ..utils import mean_pooling


class InferenceModel(nn.Module):
    def __init__(self, device, text_encoder, image_encoder,
                 video_encoder, audio_encoder, ie_model, ue_model, decoder):
        super().__init__()
        self.device = device
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.video_encoder = video_encoder  # not implemented
        self.audio_encoder = audio_encoder  # not implemented
        self.ie_model = ie_model
        self.ue_model = ue_model
        self.decoder = decoder

    def get_user_embedding(self, item_embeddings, user_embedding=None):
        return self.ue_model(item_embeddings, user_embedding)

    def get_item_embedding(self, x, content_type):
        if content_type == 'text':
            attention_mask, input_ids = x['attention_mask'], x['input_ids']
            attention_mask = attention_mask.to(self.device)
            input_ids = input_ids.to(self.device)
            item_universal_embedding = self.text_encoder(attention_mask=attention_mask,
                                                         input_ids=input_ids).last_hidden_state
            item_universal_embedding = mean_pooling(item_universal_embedding, attention_mask)
            item_universal_embedding = self.ie_model(item_universal_embedding)
        elif content_type == 'image':
            pixel_values = x['pixel_values']
            pixel_values = pixel_values.to(self.device)
            item_universal_embedding = self.image_encoder(pixel_values).last_hidden_state
            item_universal_embedding = mean_pooling(
                item_universal_embedding,
                torch.ones(item_universal_embedding.size()[:-1]).to(self.device)
            )
            item_universal_embedding = self.ie_model(item_universal_embedding)
        elif content_type == 'audio':
            item_universal_embedding = self.audio_encoder(**x).last_hidden_state
            item_universal_embedding = mean_pooling(
                item_universal_embedding,
                torch.ones(item_universal_embedding.size()[:-1]).to(self.device)
            )
            item_universal_embedding = self.ie_model(item_universal_embedding)
        elif content_type == 'video':
            item_universal_embedding = self.video_encoder(**x).last_hidden_state
            item_universal_embedding = mean_pooling(
                item_universal_embedding,
                torch.ones(item_universal_embedding.size()[:-1]).to(self.device)
            )
            item_universal_embedding = self.ie_model(item_universal_embedding)
        else:
            raise ValueError
        return item_universal_embedding

    def predict(self, item_embeddings, user_embedding):
        if isinstance(item_embeddings, list):
            item_embeddings = torch.cat(item_embeddings, dim=0)
        return self.decoder(user_embedding, item_embeddings)
