# NLP Library

## Requirement

- numpy==1.16.2
- tqdm==4.31.1
- jieba==0.39

## API

### Class Embedding

把 `EmbeddingGenerator` 生成的資料包裝成 `Embedding` 物件

```python
class Embedding(self, w2v_dict, w2id_dict, id2w_dict)
```

### Class EmbeddingGenerator

```python
class EmbeddingGenerator(dim: int = 0,special_tokens2id: dict = None,)
```

- Parameters
  - dim : 詞向量維度
  - special_tokens2id : 特殊詞的 id 對照字典, ex: {"PAD": 0, "EOS": 1}

#### load_word2vec_file(self, file_name: str = None, token_size: int = 0)

輸入詞向量檔案，產生其 word2vector dictionary, word2id dictionary, id2word dictionary，並返回 `Embedding` 物件

- Parameters
  - file_name: 詞向量檔案路徑
  - token_size: 詞的數量