import torch
import torch.nn as nn

from .base import F, C, T, nnclassifier, astag


class _cnn1d_model(nn.Module):
    '''
    Input: (B, F, T), where F as channel, T as seq_len in `torch.conv1d`
    '''

    def __init__(self, dropout):
        super(_cnn1d_model, self).__init__()

        # Extract features, 1D conv layers, full kernel
        self.features = nn.Sequential(
            nn.Conv1d(F, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        # calculate fc layer length
        self.fcs = T - 5 + 1
        self.fcs = self.fcs // 2
        self.fcs = self.fcs - 3 + 1
        self.fcs = self.fcs // 2
        self.fcs = self.fcs - 3 + 1
        self.fcs = self.fcs // 2
        self.fcs = 128 * self.fcs

        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.fcs, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, C),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out


@astag
class cnn1d(nnclassifier):

    def __init__(self,
                 lr,
                 epochs,
                 batch_size,
                 dropout):

        super(cnn1d, self).__init__(lr, epochs, batch_size)

        self.dropout = dropout
        self.lr = lr

    def flash_training(self):
        model = _cnn1d_model(self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        return model, optimizer, criterion

    def xpip(self, x):
        return x
