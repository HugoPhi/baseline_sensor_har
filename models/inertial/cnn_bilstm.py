import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        batch_size, seq_len = x.size(0), x.size(1)
        input_shape = x.shape[2:]
        reshaped_x = x.view(batch_size * seq_len, *input_shape)
        output = self.module(reshaped_x)
        output_shape = output.shape[1:]
        return output.view(batch_size, seq_len, *output_shape)


class Branch(nn.Module):
    def __init__(self, in_channels, kernel_size, dropout_rate=0.5):
        super(Branch, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            TimeDistributed(nn.Conv1d(in_channels, 64, kernel_size, padding=padding)),
            nn.ReLU(),
            TimeDistributed(nn.Conv1d(64, 32, kernel_size, padding=padding)),
            nn.ReLU(),
            TimeDistributed(nn.Dropout(dropout_rate)),
            TimeDistributed(nn.MaxPool1d(2)),
            TimeDistributed(nn.Flatten())
        )

    def forward(self, x):
        return self.conv_block(x)


class cnn_bilstm(nn.Module):
    def __init__(self, input_channels, seq_len, input_length, num_classes):
        super(cnn_bilstm, self).__init__()
        self.branch3 = Branch(input_channels, 3)
        self.branch7 = Branch(input_channels, 7)
        self.branch11 = Branch(input_channels, 11)

        self.feature_dim = 32 * (input_length // 2) * 3  # 32 channels * pooled_length * 3 branches

        self.bilstm = nn.LSTM(self.feature_dim, 64, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, 128)  # 64*2 for bidirectional
        self.bn = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, channels, input_length)
        b3 = self.branch3(x)
        b7 = self.branch7(x)
        b11 = self.branch11(x)

        merged = torch.cat([b3, b7, b11], dim=-1)

        # Bi-LSTM processing
        bilstm_out, _ = self.bilstm(merged)
        last_step = bilstm_out[:, -1, :]  # Take last timestep

        # Classifier
        fc_out = self.fc(last_step)
        bn_out = self.bn(fc_out)
        return nn.functional.softmax(self.classifier(bn_out), dim=1)
