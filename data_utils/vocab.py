import torch
from vncorenlp.vncorenlp import VnCoreNLP
from data_utils.vector import Vectors
from data_utils.vector import pretrained_aliases
from data_utils.utils import preprocess_sentence, reporthook, default_tokenizer
from collections import defaultdict, Counter
import logging
import six
import os
from tqdm import tqdm
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

def _default_unk_index():
    return 0

class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """
    def __init__(self, paths, max_size=None, min_freq=1, specials=['<pad>', "<sos>", "<eos>", "<unk>"],
                 vectors=None, unk_init=None, vectors_cache=None, tokenize_level="word"):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
        """ 

        if tokenize_level == "word":
            self.segmenter = self.load_tokenizer(".vncorenlp")
        else:
            self.segmenter = None

        self.make_vocab(paths)
        counter = self.freqs.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        self.stoi = defaultdict(_default_unk_index)
        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def make_vocab(self, paths):
        self.freqs = Counter()
        self.output_cats = set()
        self.max_sentence_length = 0
        for path in paths:
            sentences_file = open(os.path.join(path, "sents.txt"))
            sentiments_file = open(os.path.join(path, "sentiments.txt"))
        
            for sentence, sentiment in zip(sentences_file, sentiments_file):
                sentence = self.tokenizer(sentence)
                sentence = preprocess_sentence(sentence)
                self.freqs.update(sentence)
                self.output_cats.add(sentiment)
                if len(sentence) > self.max_sentence_length:
                    self.max_sentence_length = len(sentence)

        self.output_cats = list(self.output_cats)

    def tokenizer(self, sentence):
        if self.segmenter:
            return " ".join(self.segmenter.tokenize(sentence)[0])
        else:
            return default_tokenizer(sentence)

    def _encode_sentence(self, sentence):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.ones(self.max_sentence_length).long() * self.stoi["<pad>"]
        for i, token in enumerate(sentence):
            vec[i] = self.stoi[token]
        return vec

    def _decode_sentence(self, sentence_vecs: torch.Tensor):
        if sentence_vecs.dim() > 1:
            sentence_vecs = sentence_vecs.argmax(dim=-1)
        sentences = []
        for vec in sentence_vecs:
            sentences.append(" ".join([self.itos[idx] for idx in vec.tolist() if idx > 0]))

        return sentences

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

    def load_tokenizer(self, cache):
        urls = [
            "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar", 
            "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab", 
            "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr"
        ]
        if not os.path.isdir(os.path.join(cache)):
            os.makedirs(cache)
            logger.info('Downloading word segmenter ...')
            for url in urls:
                dest = os.path.join(cache, os.path.basename(url))
                if not os.path.isfile(dest) or not os.path.isdir(os.path.join(cache, os.path.basename(url))):
                    logger.info(f"Downloading {os.path.basename(url)}")
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
                        try:
                            urlretrieve(url, dest, reporthook=reporthook(t))
                        except KeyboardInterrupt as exception:  # remove the partial zip file
                            os.remove(dest)
                            raise exception

        return VnCoreNLP(os.path.join(cache, "VnCoreNLP-1.1.1.jar"), annotators="wseg", max_heap_size='-Xmx500m')

    def load_vectors(self, vectors, **kwargs):
        """
        Arguments:
            vectors: one of or a list containing instantiations of the
                GloVe, CharNGram, or Vectors classes. Alternatively, one
                of or a list of available pretrained vectors:
                fasttext.vi.300d
                phow2v.syllable.100d
                phow2v.syllable.300d
            Remaining keyword arguments: Passed to the constructor of Vectors classes.
        """
        if not isinstance(vectors, list):
            vectors = [vectors]
        for idx, vector in enumerate(vectors):
            if six.PY2 and isinstance(vector, str):
                vector = six.text_type(vector)
            if isinstance(vector, six.string_types):
                # Convert the string pretrained vector identifier
                # to a Vectors object
                if vector not in pretrained_aliases:
                    raise ValueError(
                        "Got string input vector {}, but allowed pretrained "
                        "vectors are {}".format(
                            vector, list(pretrained_aliases.keys())))
                vectors[idx] = pretrained_aliases[vector](**kwargs)
            elif not isinstance(vector, Vectors):
                raise ValueError(
                    "Got input vectors of type {}, expected str or "
                    "Vectors object".format(type(vector)))

        tot_dim = sum(v.dim for v in vectors)
        self.vectors = torch.Tensor(len(self), tot_dim)
        for i, token in enumerate(self.itos):
            start_dim = 0
            for v in vectors:
                end_dim = start_dim + v.dim
                self.vectors[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim
            assert(start_dim == tot_dim)

    def set_vectors(self, stoi, vectors, dim, unk_init=torch.Tensor.zero_):
        """
        Set the vectors for the Vocab instance from a collection of Tensors.
        Arguments:
            stoi: A dictionary of string to the index of the associated vector
                in the `vectors` input argument.
            vectors: An indexed iterable (or other structure supporting __getitem__) that
                given an input index, returns a FloatTensor representing the vector
                for the token associated with the index. For example,
                vector[stoi["string"]] should return the vector for "string".
            dim: The dimensionality of the vectors.
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
        """
        self.vectors = torch.Tensor(len(self), dim)
        for i, token in enumerate(self.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                self.vectors[i] = vectors[wv_index]
            else:
                self.vectors[i] = unk_init(self.vectors[i])