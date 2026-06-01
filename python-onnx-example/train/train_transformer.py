import argparse

from torch.utils.data import DataLoader

from transformer.transformer import Transformer
from util.text_tokenizer import TextTokenizer
from util.translation_dataset import TranslationDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train_transformer",
        description="Script to train a transformer",
    )
    parser.add_argument("--en-tokenizer", required=True)
    parser.add_argument("--en-train", required=True)
    parser.add_argument("--en-val", required=True)
    parser.add_argument("--en-test", required=True)
    parser.add_argument("--pt-tokenizer", required=True)
    parser.add_argument("--pt-train", required=True)
    parser.add_argument("--pt-val", required=True)
    parser.add_argument("--pt-test", required=True)
    args = parser.parse_args()

    en_tokenizer = TextTokenizer.from_file(path=args.en_tokenizer)
    pt_tokenizer = TextTokenizer.from_file(path=args.pt_tokenizer)

    train_ds = TranslationDataset(
        src_file=args.en_train,
        tgt_file=args.pt_train,
        src_tokenizer=en_tokenizer,
        tgt_tokenizer=pt_tokenizer,
    )
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_ds = TranslationDataset(
        src_file=args.en_val,
        tgt_file=args.pt_val,
        src_tokenizer=en_tokenizer,
        tgt_tokenizer=pt_tokenizer,
    )
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=True)
    test_ds = TranslationDataset(
        src_file=args.en_test,
        tgt_file=args.pt_test,
        src_tokenizer=en_tokenizer,
        tgt_tokenizer=pt_tokenizer,
    )
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=True)

    # tokenizers params might need tuning
    transformer = Transformer(
        source_vocab_size=en_tokenizer.vocab_size,
        target_vocab_size=pt_tokenizer.vocab_size,
        seq_length=en_tokenizer.max_len,
    )
