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
        self._num_word = len(self._w2id_dict)

        print(len(self._w2v_dict), len(self._w2id_dict))

    def __len__(self):
        return self._num_word

    @property
    def num_word(self):
        return self._num_word

    @property
    def w2v(self):
        return self._w2v_dict

    @property
    def w2id(self):
        return self._w2id_dict

    def create_id2w(self) -> dict:
        print("Get ID to Word Dictionary....")
        if not self._w2id_dict:
            raise ValueError(self._w2id_dict)

        return {idx: word for word, idx in self._w2id_dict.items()}

    def create_emb_matrix(self) -> list:
        print("Get embedding matrix.....")

        emb_matrix = np.zeros(
            (self._num_word+1, self._dim), dtype=np.float32)

        for w, i in tqdm(self._w2id_dict.items()):
            emb_matrix[i] = self._w2v_dict[w]

        return emb_matrix


class EmbeddingGenerator:
    """
    Parameters
    -----------
    - file_name : target file name
    - dim : the dimension of the vector
    - special_tokens2id : the w2id dictionary of the special tokens , ex: {"PAD": 0, "EOS": 1}
    - total_token_size : the  size of total tokens
    - random : whether randomly generate the vector
    """

    def __init__(
        self,
        file_name: str = None,
        dim: int = 0,
        total_token_size: int = 0,
        random: bool = False,
        special_tokens2id: dict = None,
    ):
        self._w2v_dict = {}
        self._w2id_dict = {}
        self._dim = dim
        self._special_tokens = None
        self._file_name = file_name
        self._total_token_size = total_token_size
        self._random = random

        if special_tokens2id:

            if not isinstance(special_tokens2id, dict):
                raise TypeError(f"special tokens need a dict....")

            self._special_tokens = special_tokens2id

    def create_embedding(self) -> Embedding:
        '''
        Create word2vector dictionary, word2id dictionary

        Return
        ------
        Embedding Object
        '''

        print('Create embedding.....')

        if not self._file_name:
            raise ValueError("no file name")

        if not self._random:
            self._create_w2v()
        else:
            self._create_random_w2v()

        self._create_w2id()

        return Embedding(w2v_dict=self._w2v_dict, w2id_dict=self._w2id_dict)

    def _read_file_line(self):
        '''
        read a line from file
        '''
        with codecs.open(self._file_name, 'r', encoding='utf-8') as f:
            next(f)

            for line in tqdm(f, total=self._total_token_size):
                yield line

    def _create_w2v(self):
        '''
        create w2v dictionary
        '''
        print("Get Word to Vector Dictionary....")
        for line in self._read_file_line():
            array = line.split()
            word = array[0].strip()
            vector = array[1:]
            if vector:
                self._w2v_dict[word] = np.array(vector, dtype=np.float32)
            else:
                raise ValueError("can not find vector")

        if self._special_tokens:
            for key in self._special_tokens.keys():
                self._w2v_dict[key] = np.zeros(self._dim,
                                               dtype=np.int32)

        print("total word:", len(self._w2v_dict))

    def _create_random_w2v(self):
        '''
        randomly create w2v dictionary
        '''

        for line in self._read_file_line():
            word = line.split()[0].strip()
            vector = np.random.normal(loc=0, scale=0.1, size=self._dim)
            self._w2v_dict[word] = vector

        if self._special_tokens:
            for key in self._special_tokens.keys():
                self._w2v_dict[key] = np.zeros(self._dim,
                                               dtype=np.int32)

    def _create_w2id(self):
        '''
        create w2id dictionary
        '''
        print("Get Word to ID Dictionary....")

        if not self._w2v_dict:
            raise ValueError("word to vector dict is not exist....")

        if self._special_tokens:

            self._w2id_dict.update(self._special_tokens)

            w2v_keys = self._w2v_dict.keys()
            special_token_keys = self._special_tokens.keys()

            for i, k in enumerate(w2v_keys,
                                  start=len(self._special_tokens)):
                if k not in special_token_keys:
                    self._w2id_dict[k] = i
        else:
            for i, k in enumerate(self._w2v_dict.keys()):
                self._w2id_dict[k] = i


if __name__ == "__main__":
    embg = EmbeddingGenerator(file_name='./embedding/Total_word.word',
                              total_token_size=1292607, dim=300, special_tokens2id={"PAD": 0, "EOS": 1})
    emb = embg.create_embedding()

    print(emb.w2v['的'])

    eg = EmbeddingGenerator(
        file_name="./embedding/chinese_character.text",
        dim=100,
        total_token_size=6722,
        random=True,
    )

    eb = eg.create_embedding()
