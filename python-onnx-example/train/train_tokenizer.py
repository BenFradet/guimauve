import argparse

from util.text_tokenizer import TextTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train_tokenizer",
        description="Script to train a tokenizer",
    )
    parser.add_argument("-i", "--inputs", required=True, nargs="+")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    tokenizer = TextTokenizer()
    tokenizer.train(args.inputs)
    tokenizer.save(args.output)
