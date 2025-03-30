import torch
import torch.nn as nn
import numpy as np

from .base import F, C, nnclassifier, astag


class _bigru_model(nn.Module):
    '''
    Input: (B, T, F), where F as channel, T as seq_len in `torch.LSTM`
    '''

    def __init__(self, hidden_dim):
        super(_bigru_model, self).__init__()

        self.gru = nn.GRU(
            input_size=F,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True  # open bidir
        )

        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(2 * hidden_dim, C)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.gru(x, h0)
        out = out[:, -1, :]

        outputs = self.fc(out)
        return outputs


@astag
class bigru(nnclassifier):
    def __init__(self,
                 lr,
                 epochs,
                 batch_size,
                 hidden_dim=32):
        super(bigru, self).__init__(lr, epochs, batch_size)

        self.hidden_dim = hidden_dim
        self.lr = lr

    def flash_training(self):
        model = _bigru_model(self.hidden_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        return model, optimizer, criterion

    def xpip(self, x):
        return np.transpose(x, (0, 2, 1))  # (B, F, T) -> (B, T, F)
