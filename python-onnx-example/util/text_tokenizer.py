from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class TextTokenizer:
    def __init__(self, vocab_size: int = 8000, max_len: int = 128) -> None:
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.enable_padding(length=max_len, pad_id=0)
        self.tokenizer.enable_truncation(max_length=max_len)

    def train(self, files: list[str]) -> None:
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[START]", "[END]"],
        )
        self.tokenizer.train(files, trainer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, encoding: list[int]) -> str:
        return self.tokenizer.decode(encoding)
