import torch
from torch import nn


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        """Basic text classification model composed of a EmbeddingBag
        layer plus a linear layer for the classification purpose.
        It takes into input a batch in the for of a
        tuple of (text, offset), where text contains the encoded
        text in a integer form, and offset:

            {
                'label':  tensor([ 3, 2, 2, 1, 1, 1, 0, 3]),
                'text':   tensor([ 3, 5, 235, ..., 3820, 130, 294]),
                'offset': tensor([ 0, 249, 369, 497, 626, 789, 1020, 1140])
            }

        Note:

            For a batch of 2 samples of text with 4 words each, we will have,
            for example, input=[1,2,4,5,4,3,2,9] and offsets=[0,4].
            See https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
            for further information.

        Args:
            vocab_size (int): size of the dictionary of text encodings
            embed_dim (int): the size of each embedding vector
            num_class (int): the number of classes to predict
        """
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(text, offset)
        return self.fc(emb)
