from torch import nn


class UserEmbeddingModel(nn.Module):
    def __init__(self, gru_input_size, gru_hidden_size, gru_num_layers, dropout):
        super().__init__()
        self.gru_input_size = gru_input_size
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.dropout = dropout
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            dropout=self.dropout
        )

    def forward(self, item_embedding_list, user_embedding=None):
        for item_embedding in item_embedding_list:
            if user_embedding is None:
                output, user_embedding = self.gru(item_embedding)
            else:
                output, user_embedding = self.gru(item_embedding, user_embedding)
        return user_embedding
