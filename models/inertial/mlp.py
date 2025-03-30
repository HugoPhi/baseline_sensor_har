import torch
import torch.nn as nn

from .base import astag, nnclassifier, F, T, C


class _mlp_model(nn.Module):
    def __init__(self, hidden_dims, dropout):
        super(_mlp_model, self).__init__()

        layers = []
        prev_dim = F * T
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, C))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


@astag
class mlp(nnclassifier):

    def __init__(self,
                 lr,
                 epochs,
                 batch_size,
                 hidden_dims,
                 dropout=0.2):

        super(mlp, self).__init__(lr, epochs, batch_size)

        self.model = _mlp_model(
            hidden_dims=hidden_dims,
            dropout=dropout
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def xpip(self, x):
        return x.reshape(-1, F * T)  # (B, F, T) -> (B, F*T)
