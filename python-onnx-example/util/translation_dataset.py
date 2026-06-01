import torch
from torch.utils.data import Dataset

from util.text_tokenizer import TextTokenizer


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_tokenizer: TextTokenizer,
        tgt_tokenizer: TextTokenizer,
    ) -> None:
        """
        Args:
            src_file: location of the file in the source language
            tgt_file: location of the file in the target language
            src_tokenizer: tokenizer to use for the source language
            tgt_tokenizer: tokenizer to use for the target language
        """
        with open(src_file) as f:
            src_lines = f.readlines()
        with open(tgt_file) as f:
            tgt_lines = f.readlines()

        src_encoded = [src_tokenizer.encode(line.strip()) for line in src_lines]
        self.src = torch.tensor(src_encoded, dtype=torch.long)
        tgt_encoded = [tgt_tokenizer.encode(line.strip()) for line in tgt_lines]
        self.tgt = torch.tensor(tgt_encoded, dtype=torch.long)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]
