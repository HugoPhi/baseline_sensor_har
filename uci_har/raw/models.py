import torch as tc
from torch import nn

'''
mlp
'''


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


'''
cnns
'''


class Conv1d_3x3(nn.Module):
    def __init__(self, input_width, input_height, input_channels, output_dim, dropout=0.5):
        super(Conv1d_3x3, self).__init__()

        layers = []
        layers_config = [
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 32, "use_bn": True, "padding": 1},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 64, "use_bn": True, "padding": 1},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 128, "use_bn": False, "padding": 1}
        ]

        wd, he = input_width, input_height
        prev_filters = input_channels
        for layer_config in layers_config:
            kernel_size = layer_config["kernel_size"]
            pool_size = layer_config["pool_size"]
            filters = layer_config["filters"]
            use_bn = layer_config["use_bn"]
            padding = layer_config["padding"]

            layers.append(nn.Conv2d(prev_filters, filters, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU())

            if use_bn:
                layers.append(nn.BatchNorm2d(filters))

            layers.append(nn.MaxPool2d(pool_size))

            layers.append(nn.Dropout(dropout))

            # W/H here is width/height, they are the same
            # Conv      : W <- (W + 2P - K) // S + 1
            # MaxPooling: W <- W // 2
            he = (he - kernel_size[0] + 2 * padding + 1) // pool_size[0]
            wd = (wd - kernel_size[1] + 2 * padding + 1) // pool_size[1]
            prev_filters = filters

        layers.append(nn.Flatten())
        layers.append(nn.Linear(prev_filters * wd * he, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Conv2d_3x3(nn.Module):
    def __init__(self, input_width, output_dim, input_channels, dropout=0.5):
        super(Conv2d_3x3, self).__init__()

        layers_config = [
            {"kernel_size": (3,), "pool_size": (2,), "filters": 32, "use_bn": True, "padding": 1},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 64, "use_bn": True, "padding": 1},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 128, "use_bn": False, "padding": 1}
        ]

        layers = []
        wd = input_width
        prev_filters = input_channels
        for layer_config in layers_config:
            kernel_size = layer_config["kernel_size"]
            pool_size = layer_config["pool_size"]
            filters = layer_config["filters"]
            use_bn = layer_config["use_bn"]
            padding = layer_config.get("padding", 0)

            layers.append(nn.Conv1d(prev_filters, filters, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU())

            if use_bn:
                layers.append(nn.BatchNorm1d(filters))

            layers.append(nn.MaxPool1d(pool_size))

            layers.append(nn.Dropout(dropout))

            # 计算新的宽度
            wd = (wd - kernel_size[0] + 2 * padding + 1) // pool_size[0]
            prev_filters = filters

        layers.append(nn.Flatten())
        layers.append(nn.Linear(prev_filters * wd, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


'''
rnns
'''


class BasicRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(BasicRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          batch_first=True,
                          dropout=dropout)
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = tc.zeros(1, x.size(0), self.hidden_dim).to(x.device)  # 初始化隐藏状态
        out, _ = self.rnn(x.float(), h0)
        out = out[:, -1, :]  # 只取序列的最后一个时间步的输出
        outputs = self.fc(out)
        return outputs
