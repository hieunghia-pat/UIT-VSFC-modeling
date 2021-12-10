from torch import nn

from data_utils.vocab import Vocab

class Embedding(nn.Module):
    def __init__(self, vocab: Vocab, embedding_dim: int):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(len(vocab.stoi), embedding_dim, padding_idx=vocab.stoi["<pad>"])
        if vocab.vectors is not None:
            self.embedding.from_pretrained(vocab.vectors)

    def forward(self, x):
        return self.embedding(x)
