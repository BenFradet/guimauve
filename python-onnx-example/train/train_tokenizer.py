import argparse

from util.text_tokenizer import TextTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train_tokenizer",
        description="Script to train a tokenizer",
    )
    parser.add_argument("-i", "--inputs", required=True, nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=8000)
    args = parser.parse_args()

    tokenizer = TextTokenizer(vocab_size=args.vocab_size, max_seq_len=args.max_seq_len)
    tokenizer.train(args.inputs)
    tokenizer.save(args.output)
