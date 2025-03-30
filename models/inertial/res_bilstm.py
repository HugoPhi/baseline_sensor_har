'''
Res-BiLSTM:
    Zhao, Y., Yang, R., Chevalier, G., Xu, X., & Zhang, Z. (2018). Deep Residual Bidir-LSTM for Human Activity Recognition Using Wearable Sensors. Mathematical Problems in Engineering, 2018, 1–13. https://doi.org/10.1155/2018/7316954
'''

import torch
import torch.nn as nn
import numpy as np

from .base import F, C, nnclassifier, astag


class _res_bilstm_block(nn.Module):
    '''
    Input: (B, T, F), where F as channel, T as seq_len in `torch.LSTM`
    '''

    def __init__(self, hidden_dim, ix):
        super(_res_bilstm_block, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=F if ix == 0 else 2 * hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True  # open bidir
        )
        self.lstm2 = nn.LSTM(
            input_size=2 * hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True  # open bidir
        )
        self.hidden_dim = hidden_dim
        self.ln = nn.LayerNorm(2 * hidden_dim)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(2, x.size(0), self.hidden_dim).to(x.device)
        x, _ = self.lstm1(x, (h0, c0))
        x = nn.functional.relu(x)

        h0 = torch.zeros(2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(2, x.size(0), self.hidden_dim).to(x.device)
        out2, _ = self.lstm2(x, (h0, c0))
        out2 = nn.functional.relu(out2)

        x = x + out2
        x = self.ln(x)
        return x


class _res_bilstm_model(nn.Module):
    '''
    Input: (B, T, F), where F as channel, T as seq_len in `torch.LSTM`
    '''

    def __init__(self, hidden_dim, block_num):
        super(_res_bilstm_model, self).__init__()

        self.blocks = nn.ModuleList(
            [_res_bilstm_block(hidden_dim, ix) for ix in range(block_num)]
        )

        self.fc = nn.Linear(2 * hidden_dim, C)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = x[:, -1, :]

        outputs = self.fc(x)
        return outputs


@astag
class res_bilstm(nnclassifier):

    def __init__(self,
                 lr,
                 epochs,
                 batch_size,
                 block_num=2,
                 l2=0.005,
                 hidden_dim=32):

        super(res_bilstm, self).__init__(lr, epochs, batch_size)

        self.hidden_dim = hidden_dim
        self.block_num = block_num
        self.lr = lr
        self.l2 = l2

    def flash_training(self):
        model = _res_bilstm_model(
            hidden_dim=self.hidden_dim,
            block_num=self.block_num
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2)

        return model, optimizer, criterion

    def xpip(self, x):
        return np.transpose(x, (0, 2, 1))  # (B, F, T) -> (B, T, F)
