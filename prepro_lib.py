import numpy as np
import jieba
from tqdm import tqdm
import pickle
import codecs


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


class Embedding:
    def __init__(self, w2v_dict: dict, w2id_dict):
        self._w2v_dict = w2v_dict
        self._w2id_dict = w2id_dict
        self._num_word = len(self._w2v_dict)

    def __len__(self):
        return self._num_word

    @property
    def w2v(self):
        return self._w2v_dict

    @property
    def w2id(self):
        return self._w2id_dict

    def get_id2w(self) -> dict:
        print("Get ID to Word Dictionary....")
        if not self._w2id_dict:
            raise ValueError(self._w2id_dict)

        return {idx: word for word, idx in self._w2id_dict.items()}

    def get_emb_matrix(self) -> list:
        print("Get embedding matrix.....")

        emb_matrix = np.zeros(
            (self._num_word+1, self._dim), dtype=np.float32)

        for w, i in tqdm(self._w2id_dict.items()):
            emb_matrix[i] = self._w2v_dict[w]

        return emb_matrix


class EmbeddingGenerator:
    def __init__(
        self,
        file_name: str = None,
        dim: int = 0,
        total_token_size: int = 0,
        special_tokens2id: dict = None,
    ):
        self._w2v_dict = {}
        self._w2id_dict = {}
        self._num_word = 0
        self._dim = dim
        self._special_tokens = None
        self._file_name = file_name
        self._total_token_size = total_token_size

        if special_tokens2id:

            if not isinstance(special_tokens2id, dict):
                raise TypeError(f"special tokens need a dict....")

            self._special_tokens = special_tokens2id

    @property
    def num_word(self):
        return self._num_word

    def load_word2vec_file(self):

        print('Load word to vector file.....')

        if not self._file_name:
            raise ValueError("no file name")

        self._get_w2v()
        self._get_w2id()

        return Embedding(w2v_dict=self._w2v_dict, w2id_dict=self._w2id_dict)

    def _read_file_line(self):
        with codecs.open(self._file_name, 'r', encoding='utf-8') as f:
            next(f)

            for line in tqdm(f, total=self._total_token_size):
                yield line

    def _get_w2v(self):
        print("Get Word to Vector Dictionary....")
        for line in self._read_file_line():
            array = line.split()
            word = array[0]
            vector = np.array(array[1:], dtype=np.float32)

            self._w2v_dict[word] = vector

        if self._special_tokens:
            for key in self._special_tokens.keys():
                self._w2v_dict[key] = np.zeros(self._dim,
                                               dtype=np.int32)

        self._num_word = len(self._w2v_dict)
        print("total word:", self._num_word)

    def _get_w2id(self):
        print("Get Word to ID Dictionary....")

        if not self._w2v_dict:
            raise ValueError("word to vector dict is not exist....")

        if self._special_tokens:
            self._w2id_dict.update(self._special_tokens)

        for i, k in enumerate(self._w2v_dict.keys(),
                              start=len(self._special_tokens)):
            if k not in self._special_tokens.keys():
                self._w2id_dict[k] = i


if __name__ == "__main__":
    embg = EmbeddingGenerator(file_name='./embedding/Total_word.word',
                              total_token_size=1292608, dim=300, special_tokens2id={"PAD": 0, "EOS": 1})
    emb = embg.load_word2vec_file()

    print(emb.w2v['我'])
