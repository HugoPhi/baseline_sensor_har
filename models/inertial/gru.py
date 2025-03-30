import torch
import torch.nn as nn
import numpy as np

from .base import F, C, nnclassifier, astag


class _gru_model(nn.Module):
    '''
    Input: (B, T, F), where F as channel, T as seq_len in `torch.LSTM`
    '''

    def __init__(self, hidden_dim):
        super(_gru_model, self).__init__()

        self.gru = nn.GRU(
            input_size=F,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(self.hidden_dim, C)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.gru(x, h0)
        out = out[:, -1, :]

        outputs = self.fc(out)
        return outputs


@astag
class gru(nnclassifier):
    def __init__(self,
                 lr,
                 epochs,
                 batch_size,
                 hidden_dim=32):
        super(gru, self).__init__(lr, epochs, batch_size)

        self.hidden_dim = hidden_dim
        self.lr = lr

    def flash_training(self):
        model = _gru_model(self.hidden_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        return model, optimizer, criterion

    def xpip(self, x):
        return np.transpose(x, (0, 2, 1))  # (B, F, T) -> (B, T, F)
