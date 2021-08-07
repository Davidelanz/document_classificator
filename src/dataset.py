import pandas as pd
import torch
import torchtext
from torch.utils.data import Dataset


class DocTextDataset(Dataset):
    """This dataset takes as input a dataframe containing a string column named "text"
    and a (categorical) string column named "label". Optionally, the code2label
    dictionary can be passed, if not, categorical encoding will be performed
    automatically on the passed fataframe.

    Args:
        df (pd.DataFrame): The dataframe containing "text" and "label" colums
        code2label (dict, optional): Pre-defined categorical encoding. Defaults to None.
    """

    def __init__(self, df: pd.DataFrame, code2label: dict, vocab, tokenizer) -> None:
        self.df = df
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.code2label = dict(enumerate(df['label'].astype("category").cat.categories))
        self.label2code = {v: k for k, v in code2label.items()}
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        self.label_pipeline = lambda x: self.label2code[x]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.label_pipeline(self.df.iloc[idx]["label"])
        text = self.text_pipeline(self.df.iloc[idx]["text"])
        sample = {
            "text": torch.LongTensor(text),
            "label": torch.LongTensor([label])
        }
        return sample
