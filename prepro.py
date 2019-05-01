import numpy as np
import jieba
from opencc import OpenCC
from tqdm import tqdm
import pickle
import codecs
from traceback import print_exc


def jieba_cut(text: str) -> list:
    '''
    return token list of one sentence

    Parameters
    ----------
    text: a sentence or string which need to tokenize

    '''
    return [w for w in jieba.cut(text)]


def tokenize_sentences(sentences: list) -> list:
    '''
    return token lists of sentences

    Parameters
    ----------
    sentences : sentences which need to tokenize
    '''

    return [jieba_cut(s) for s in sentences]


def save(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def cc(mode: str):
    return OpenCC(mode)


class Embedding:
    def __init__(self, w2v_dict, w2id_dict, id2w_dict):
        self.w2v_dict = w2id_dict
        self.w2id_dict = w2id_dict
        self.id2w_dict = id2w_dict

    def __len__(self):
        return len(self.w2v_dict)


class EmbeddingGenerator:
    def __init__(
        self,
        dim: int = 0,
        special_tokens: dict = None,
    ):
        self._w2v_dict = {}
        self._w2id_dict = {}
        self._id2w_dict = None
        self._num_word = 0
        self._dim = dim
        self._special_tokens = None

        if special_tokens:

            if not isinstance(special_tokens, dict):
                print('we need dict')
                raise TypeError(special_tokens)

            self._special_tokens = special_tokens

    def load_word2vec_file(self, file_name: str = None, token_size: int = 0):

        print('Load word to vector file.....')

        if not file_name:
            raise ValueError(file_name)

        self._get_w2v(file_name, token_size)
        self._get_w2id()
        self._get_id2w()

        return Embedding(
            w2v_dict=self._w2v_dict,
            w2id_dict=self._w2id_dict,
            id2w_dict=self._id2w_dict
        )

    def _read_file_line(self, file_name: str, token_size: int):
        with codecs.open(file_name, 'r', encoding='utf-8') as f:
            next(f)

            for line in tqdm(f, total=token_size):
                yield line

    def _get_w2v(self, file_name: str, token_size: int):
        print("Get Word to Vector Dictionary....")
        for line in self._read_file_line(file_name, token_size):
            array = line.split()
            word = array[0]
            vector = np.array([float(val) for val in array[1:]])

            self._w2v_dict[word] = vector
            self._num_word += 1

        if self._special_tokens:
            for key in self._special_tokens.keys():
                self._w2v_dict[key] = np.zeros(self._dim)

    def _get_w2id(self):
        print("Get Word to ID Dictionary....")
        if not self._w2v_dict:
            raise ValueError(self._w2v_dict)

        if self._special_tokens:
            self._w2id_dict.update(self._special_tokens)

        for i, k in enumerate(self._w2v_dict.keys(), start=self._num_word):
            self._w2id_dict[k] = i

    def _get_id2w(self):
        print("Get ID to Word Dictionary....")
        if not self._w2id_dict:
            raise ValueError(self._w2id_dict)

        self._id2w_dict = {idx: word for word, idx in self._w2id_dict.items()}


if __name__ == "__main__":
    embg = EmbeddingGenerator(dim=300, special_tokens={"PAD": 0, "EOS": 1})
    emb = embg.load_word2vec_file('./embedding/Total_word.word', 1292608)