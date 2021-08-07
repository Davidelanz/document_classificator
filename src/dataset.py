from typing import Callable

import pandas as pd
import torch
import torchtext
from torch.utils.data import Dataset


class DocTextDataset(Dataset):
    """This dataset takes as input a dataframe containing labelled text
    along with a defined label encoding dictionari and data-specific
    torchtext's tokenizer & vocabulary, returning a dictionary containing
    the encoded text and label in the form:

        {
            'text': tensor([0, 1585, 88, ..., 16]),
            'label': tensor([0])
        }


    Args:
        df (pd.DataFrame): The dataframe containing a string column named
            "text" and a (categorical) string column named "label"
        code2label (dict[int, str]): Pre-defined categorical encoding
            (from integers to string labels)
    """

    def __init__(self, df: pd.DataFrame, code2label: "dict[int, str]",
                 vocab: torchtext.vocab.Vocab, tokenizer: Callable) -> None:
        self.df = df
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.code2label = dict(
            enumerate(df['label'].astype("category").cat.categories))
        self.label2code = {v: k for k, v in code2label.items()}
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        self.label_pipeline = lambda x: self.label2code[x]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> "dict[str, torch.Tensor]":
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.label_pipeline(self.df.iloc[idx]["label"])
        text = self.text_pipeline(self.df.iloc[idx]["text"])
        sample = {
            "text": torch.LongTensor(text),
            "label": torch.LongTensor([label])
        }
        return sample
