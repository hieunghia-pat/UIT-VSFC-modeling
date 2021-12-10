import torch
from torch.utils.data import Dataset, DataLoader
from data_utils.utils import preprocess_sentence
from data_utils.vocab import Vocab
import os
import config

class Dataset(Dataset):
    """ VQA dataset, open-ended """
    def __init__(self, path, vocab=None):
        super(Dataset, self).__init__()
        self.load_dataset(path)

        self.vocab = Vocab([path]) if vocab is None else vocab

    @property
    def num_tokens(self):
        return len(self.vocab.stoi)

    def load_dataset(self, path):
        sentences_file = open(os.path.join(path, "sents.txt"))
        sentiments_file = open(os.path.join(path, "sentiments.txt"))

        self.data = []
        for sentence, sentiment in zip(sentences_file, sentiments_file):
            self.data.append({
                "sentence": preprocess_sentence(sentence),
                "sentiment": sentiment
            })

    def __getitem__(self, idx):
        sentence = self.data[idx]["sentence"]
        sentiment = self.data[idx]["sentiment"]

        sentence = self.vocab._encode_sentence(sentence)

        return sentence, torch.tensor(int(sentiment))

    def __len__(self):
        return len(self.data)

    def get_loader(self):
        loader = DataLoader(
            self,
            batch_size = config.batch_size,
            shuffle=True
        )

        return loader