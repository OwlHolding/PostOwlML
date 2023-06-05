from torch import nn


class UserEmbeddingModel(nn.Module):
    def __init__(self, item_embedding_model, gru_input_size, gru_hidden_size, gru_num_layers, dropout, device):
        super().__init__()
        self.item_embedding_model = item_embedding_model
        self.gru_input_size = gru_input_size
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.dropout = dropout
        self.device = device
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            dropout=self.dropout
        )

    def forward(self, input_ids_list, attention_mask_list, batch_size, old_user_embedding=None):
        user_embedding = self.init_hidden(batch_size)
        for i in range(len(input_ids_list)):
            item_embedding = self.item_embedding_model(input_ids_list[i], attention_mask_list[i])
            if old_user_embedding is not None:
                user_embedding = self.gru(item_embedding, old_user_embedding)
                old_user_embedding = None
            else:
                user_embedding = self.gru(item_embedding, user_embedding)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
