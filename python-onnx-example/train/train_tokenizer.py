import argparse

from util.text_tokenizer import TextTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    tokenizer = TextTokenizer()
    tokenizer.train([args.input])
    tokenizer.save(args.output)
