import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.types as types
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformer.transformer import Transformer
from util.text_tokenizer import TextTokenizer
from util.translation_dataset import TranslationDataset


def compute_loss(
    predictions: torch.Tensor, label: torch.Tensor, pad_idx: int = 0
) -> tuple[torch.Tensor, types.Number, types.Number]:
    """
    pad is '[PAD]' at index 0
    """
    # label is [batch size, seq len]
    # this turns it into [batch size * seq len]
    label = label.contiguous().view(-1)
    # output of transofmer is [batch size, seq len, vocab size]
    # this turns it into [batch size * seq len, vocab size]
    predictions = predictions.view(-1, predictions.size(-1))

    loss = F.cross_entropy(predictions, label, ignore_index=pad_idx, reduction="sum")

    # predicted token id with highest score
    predictions = predictions.argmax(dim=1)

    non_pad_mask = label.ne(pad_idx)

    correct_words = predictions.eq(label).masked_select(non_pad_mask).sum().item()
    words = non_pad_mask.sum().item()

    return loss, words, correct_words


def epoch(
    model: nn.Module,
    data_loader: DataLoader,
    scheduler: optim.lr_scheduler.LRScheduler,
    training: bool,
) -> tuple[float, float]:
    model.train() if training else model.eval()
    total_loss, total_words, total_correct_words = 0, 0, 0

    desc = "training" if training else "validation"

    with torch.set_grad_enabled(training):
        for source, target in tqdm(
            data_loader,
            mininterval=2,
            desc=desc,
            leave=False,
        ):
            # input to the decoder [START, t1, t2, ...]
            target_input = target[:, :-1]
            # expected output [t1, t2, ..., END]
            target_label = target[:, 1:]

            if training:
                scheduler.optimizer.zero_grad()

            predictions = model(source, target_input)

            loss, words, correct_words = compute_loss(predictions, target_label)

            if training:
                loss.backward()
                scheduler.optimizer.step()
                # changes the learning rate
                scheduler.step()

            total_words += words
            total_correct_words += correct_words
            total_loss += loss.item()

    loss_per_word = total_loss / total_words
    accuracy = total_correct_words / total_words
    return (loss_per_word, accuracy)


def debug(
    stage: str,
    start_time: float,
    loss: float,
    accuracy: float,
    learning_rate: float | None = None,
) -> None:
    elapsed = (time.time() - start_time) / 60
    perplexity = math.exp(min(loss, 100))
    lr_str = (
        f", learning rate: {learning_rate:8.5f}" if learning_rate is not None else ""
    )
    print(
        f"{stage}, elapsed: {elapsed:3.3f}min - perplexity: {perplexity:8.5f}, accuracy: {accuracy:3.3f}{lr_str}"
    )


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
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=100000)
    parser.add_argument("--num-warmup-steps", type=int, default=4000)
    parser.add_argument("--output-dir", default="/tmp/")
    parser.add_argument("--debug", type=bool, default=True)
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
        model_dim=args.model_dim,
    )

    # taken from the paper
    optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9)
    # lr increases linearly during warmup and decreases afterwards proportionnally
    # to the inverse sqrt of the step number
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda num_steps: (
            args.model_dim**-0.5
            * min(
                max(num_steps, 1) ** -0.5,
                max(num_steps, 1) * args.num_warmup_steps**-1.5,
            )
        ),
    )

    summary_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    train_log_file = os.path.join(args.output_dir, "train.log")
    val_log_file = os.path.join(args.output_dir, "validation.log")
    with open(train_log_file, "w") as train_log, open(val_log_file, "w") as val_log:
        train_log.write("epoch,loss,perplexity,accuracy\n")
        val_log.write("epoch,loss,perplexity,accuracy\n")

    for epoch_i in range(args.num_steps):
        print(f"epoch {epoch_i}")

        start_time = time.time()
        train_loss, train_accuracy = epoch(
            transformer, train_dl, scheduler, training=True
        )
        if args.debug:
            current_learning_rate = float(scheduler.get_last_lr()[0])
            debug(
                stage="Training",
                start_time=start_time,
                loss=train_loss,
                accuracy=train_accuracy,
                learning_rate=current_learning_rate,
            )

        start_time = time.time()
        val_loss, val_accuracy = epoch(transformer, val_dl, scheduler, training=False)
        if args.debug:
            debug(
                stage="Validation",
                start_time=start_time,
                loss=val_loss,
                accuracy=val_accuracy,
            )
