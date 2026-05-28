from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class TextTokenizer:
    """
    byte-pair encoding subword tokenizer with padding and truncation
    must call train before encode/decode
    c.f. https://github.com/huggingface/tokenizers
    """

    def __init__(self, vocab_size: int = 8000, max_len: int = 128) -> None:
        """
        Args:
            vocab_size: maximum number of tokens in the vocabulary
            max_len: sequence length for padding and truncation
        """
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.enable_padding(length=max_len, pad_id=0)
        self.tokenizer.enable_truncation(max_length=max_len)

    def train(self, files: list[str]) -> None:
        """
        trains the tokenizer on the given text files

        Args:
            files: paths to plain text files, one sentence per line
        """
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[START]", "[END]"],
        )
        self.tokenizer.train(files, trainer)

    def encode(self, text: str) -> list[int]:
        """
        Args:
            text: input string to tokenize

        Returns:
            token IDs padded/truncated to max_len
        """
        return self.tokenizer.encode(text).ids

    def decode(self, encoding: list[int]) -> str:
        """
        Args:
            encoding: list of token IDs

        Returns:
            decoded string
        """
        return self.tokenizer.decode(encoding)

    def save(self, path: str) -> None:
        """
        saves a trained tokenizer to disk

        Args:
            path: file where the serialized tokenizer will be saved
        """
        self.tokenizer.save(path)

    @classmethod
    def from_file(cls, path: str, max_len: int = 128) -> "TextTokenizer":
        """
        creates a TextTokenizer from a file location on disk
        
        Args:
            path: file where the serialized tokenizer is saved
            max_len: sequence length for padding and truncation

        Returns:
            A trained TextTokenizer loaded from disk
        """
        tokenizer = Tokenizer.from_file(path)
        instance = cls(vocab_size=tokenizer.get_vocab_size(), max_len=max_len)
        instance.tokenizer = tokenizer
        return instance
