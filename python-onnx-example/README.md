For `transformer`, see:
- attention is all you need: https://arxiv.org/pdf/1706.03762
- https://www.tensorflow.org/text/tutorials/transformer
- https://happystrongcoder.substack.com/p/transformer-with-code-part-i-positional
- https://happystrongcoder.substack.com/p/transformer-with-code-part-ii-encoder

## Training

### Tokenizer

```bash
uv run -m train.train_tokenizer \
    -i ../datasets/pt_to_en/en.train \
    -o ../models/pt_to_en/en_tokenizer.json
```
