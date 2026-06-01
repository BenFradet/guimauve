import argparse

from util.text_tokenizer import TextTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train_transformer",
        description="Script to train a transformer",
    )
    parser.add_argument("--en", required=True)
    parser.add_argument("--pt", required=True)
    args = parser.parse_args()

    en_tokenizer = TextTokenizer.from_file(path=args.en)
    pt_tokenizer = TextTokenizer.from_file(path=args.pt)
