For `transformer`, see:
- attention is all you need: https://arxiv.org/pdf/1706.03762
- https://www.tensorflow.org/text/tutorials/transformer
- https://happystrongcoder.substack.com/p/transformer-with-code-part-i-positional
- https://happystrongcoder.substack.com/p/transformer-with-code-part-ii-encoder
- https://github.com/jadore801120/attention-is-all-you-need-pytorch

## Training

### Tokenizer

```bash
uv run -m train.train_tokenizer \
    -i ../datasets/pt_to_en/en.train \
    -o ../models/pt_to_en/en_tokenizer.json
```

### Transformer

```bash
uv run -m train.train_transformer \
    --en-tokenizer ../models/pt_to_en/en_tokenizer.json \
    --en-train ../datasets/pt_to_en/en.train \
    --en-val ../datasets/pt_to_en/en.dev \
    --en-test ../datasets/pt_to_en/en.test \
    --pt-tokenizer ../models/pt_to_en/pt_tokenizer.json \
    --pt-train ../datasets/pt_to_en/pt.train \
    --pt-val ../datasets/pt_to_en/pt.dev \
    --pt-test ../datasets/pt_to_en/pt.test \
    --output-dir ../models/pt_to_en \
    --debug
```

## Todo

- https://huggingface.co/blog/atharv6f/autoregressive-loop
