# NLP Library

## Requirement

- numpy==1.16.2
- tqdm==4.31.1
- jieba==0.39

## API

### Class Embedding

把 `EmbeddingGenerator` 生成的 word2vector 和 word2id 包裝成 `Embedding` 物件

並自行決定是否產生 embedding matrix

```python
class Embedding(self, w2v_dict, w2id_dict, id2w_dict)
```

### Class EmbeddingGenerator

```python
class EmbeddingGenerator(
        file_name: str = None,
        dim: int = 0,
        total_token_size: int = 0,
        random: bool = False,
        special_tokens2id: dict = None,
    )
```

- Parameters
  - file_name : 檔案名稱
  - dim : 詞向量維度
  - special_tokens2id : 特殊詞的 id 對照字典, ex: {"PAD": 0, "EOS": 1}
  - total_token_size : token 總數
  - random : 向量是否要用隨機

#### create_embedding()

輸入詞向量檔案，產生其 word2vector dictionary, word2id dictionary, 並返回 `Embedding` 物件
