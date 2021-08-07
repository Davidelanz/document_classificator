import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchtext


def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)


def get_vocab(text_iterator):
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    vocab = torchtext.vocab.build_vocab_from_iterator(
        yield_tokens(text_iterator, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab, tokenizer


def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # shuffle df
    df = df.sample(frac=1, random_state=42)
    # split it
    train_df, val_df, test_df = np.split(
        df, [int(.6*len(df)), int(.8*len(df))])
    return train_df, val_df, test_df


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch):
    device = get_device()
    label_list, text_list, offsets = [], [], [0]
    for sample in batch:
        _label = sample["label"]
        _text = sample["text"]
        label_list.append(_label)
        processed_text = torch.LongTensor(_text)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return {
        "label": label_list.to(device),
        "text": text_list.to(device),
        "offset": offsets.to(device)
    }


def train(model: torch.nn,
          dataloader: torch.utils.data.DataLoader,
          criterion,
          optimizer: torch.optim.Optimizer,
          epoch: int,
          log_interval: int = 500):
    model.train()
    total_acc, total_count = 0, 0
    for idx, batch in enumerate(dataloader):
        label = batch["label"]
        text = batch["text"]
        offset = batch["offset"]
        optimizer.zero_grad()
        predited_label = model(text, offset)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0


def evaluate(model: torch.nn,
             dataloader: torch.utils.data.DataLoader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            label = batch["label"]
            text = batch["text"]
            offset = batch["offset"]
            predited_label = model(text, offset)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count
