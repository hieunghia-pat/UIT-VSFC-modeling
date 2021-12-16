import torch
from torch import nn

from data_utils.vocab import Vocab
from model.embedding import Embedding

class BiLSTM(nn.Module):
    def __init__(self, vocab: Vocab, embedding_dim: int, hidden_size: int, dropout: float=0.5):
        super(BiLSTM, self).__init__()

        self.embedding = Embedding(vocab, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(in_features=2*hidden_size, out_features=len(vocab.output_cats))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        _, (x, _) = self.lstm(x)
        x = torch.cat([x[0], x[1]], dim=-1)
        x = self.dropout(self.proj(x))

        return x

class DeeperBiLSTM(nn.Module):
    def __init__(self, vocab: Vocab, embedding_dim: int, hidden_size: int, dropout: float=0.5):
        super(DeeperBiLSTM, self).__init__()

        self.embedding = Embedding(vocab, embedding_dim)
        self.lstm_1 = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(in_features=2*hidden_size, out_features=len(vocab.output_cats))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x, (h, c) = self.lstm_1(x)
        _, (x, _) = self.lstm_2(x, (h, c))
        x = torch.cat([x[0], x[1]], dim=-1)
        x = self.dropout(self.proj(x))

        return x