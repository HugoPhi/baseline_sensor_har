import torch.nn as nn
import torch as tc

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
-----------------------------

- Conv1d_3x_1:
    conv1d(3, 32, 1) -> ReLU
    max_pool(2)

    fc(flattened, 128)
    Dropout
    fc(128, 6)

- Conv1d_3x_3:
    conv1d(3, 32, 1) -> BN -> ReLU
    max_pool(2)
    conv1d(3, 64, 1) -> BN -> ReLU
    max_pool(2)
    conv1d(3, 128, 1) -> BN -> ReLU
    max_pool(2)

    fc(flattened, 128)
    Dropout
    fc(128, 6)

- Conv1d_3x_huge:
    conv1d(3, 32, 1) -> BN -> ReLU
    max_pool(2)
    conv1d(3, 32, 1) -> BN -> ReLU
    max_pool(2)
    conv1d(3, 32, 1) -> BN -> ReLU
    max_pool(2)
    conv1d(3, 32, 1) -> BN -> ReLU
    max_pool(2)
    conv1d(3, 64, 1) -> BN -> ReLU
    max_pool(2)
    conv1d(3, 64, 1) -> BN -> ReLU
    max_pool(2)
    conv1d(3, 64, 1) -> BN -> ReLU
    max_pool(2)
    conv1d(3, 64, 1) -> BN -> ReLU
    max_pool(2)
    conv1d(3, 128, 1) -> BN -> ReLU
    max_pool(2)
    conv1d(3, 128, 1) -> BN -> ReLU
    max_pool(2)

    fc(flattened, 128)
    Dropout
    fc(128, 6)

- Conv2d_3x3_1:
    conv1d(3x3, 32, 1) -> ReLU
    max_pool(2x2)

    fc(flattened, 128)
    Dropout
    fc(128, 6)

- Conv2d_3x3_3:
    conv1d(3x3, 32, 1) -> BN -> ReLU
    max_pool(2x2)
    conv1d(3x3, 64, 1) -> BN -> ReLU
    max_pool(2x2)
    conv1d(3x3, 128, 1) -> BN -> ReLU
    max_pool(2x2)

    fc(flattened, 128)
    Dropout
    fc(128, 6)

- Conv2d_3x3_huge:
    conv1d(3x3, 32, 1) -> BN -> ReLU
    max_pool(2x2)
    conv1d(3x3, 32, 1) -> BN -> ReLU
    max_pool(2x2)
    conv1d(3x3, 32, 1) -> BN -> ReLU
    max_pool(2x2)
    conv1d(3x3, 32, 1) -> BN -> ReLU
    max_pool(2x2)
    conv1d(3x3, 64, 1) -> BN -> ReLU
    max_pool(2x2)
    conv1d(3x3, 64, 1) -> BN -> ReLU
    max_pool(2x2)
    conv1d(3x3, 64, 1) -> BN -> ReLU
    max_pool(2x2)
    conv1d(3x3, 64, 1) -> BN -> ReLU
    max_pool(2x2)
    conv1d(3x3, 128, 1) -> BN -> ReLU
    max_pool(2x2)
    conv1d(3x3, 128, 1) -> BN -> ReLU
    max_pool(2x2)

    fc(flattened, 128)
    Dropout
    fc(128, 6)

'''


class Conv2d_3x3_huge(nn.Module):
    def __init__(self, input_width, input_height, input_channels, output_dim, dropout=0.5):
        super(Conv2d_3x3_huge, self).__init__()

        layers = []
        layers_config = [
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 32, "use_bn": True, "padding": 1},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 32, "use_bn": True, "padding": 1},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 32, "use_bn": True, "padding": 1},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 32, "use_bn": True, "padding": 1},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 64, "use_bn": True, "padding": 1},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 64, "use_bn": True, "padding": 1},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 64, "use_bn": True, "padding": 1},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 64, "use_bn": True, "padding": 1},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 128, "use_bn": True, "padding": 1},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 128, "use_bn": True, "padding": 1}
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
            if use_bn:
                layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU())

            layers.append(nn.MaxPool2d(pool_size))

            # W/H here is width/height, they are the same
            # Conv      : W <- (W + 2P - K) // S + 1
            # MaxPooling: W <- W // 2
            he = (he - kernel_size[0] + 2 * padding + 1) // pool_size[0]
            wd = (wd - kernel_size[1] + 2 * padding + 1) // pool_size[1]
            prev_filters = filters

        layers.append(nn.Flatten())
        layers.append(nn.Linear(prev_filters * wd * he, 128))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Conv2d_3x3_1(nn.Module):
    def __init__(self, input_width, input_height, input_channels, output_dim, dropout=0.5):
        super(Conv2d_3x3_1, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(32 * (input_width // 2) * (input_height // 2), 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)

        return x


class Conv2d_3x3_3(nn.Module):
    def __init__(self, input_width, input_height, input_channels, output_dim, dropout=0.5):
        super(Conv2d_3x3_3, self).__init__()

        # 第一层卷积：input_channels -> 32
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)

        # 第二层卷积：32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(64)

        # 第三层卷积：64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(128)

        # 全连接层
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * (input_width // 8) * (input_height // 8), 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)

        return x


class Conv1d_3x_huge(nn.Module):
    def __init__(self, input_width, output_dim, input_channels, dropout=0.5):
        super(Conv1d_3x_huge, self).__init__()

        layers_config = [
            {"kernel_size": (3,), "pool_size": (2,), "filters": 32, "use_bn": True, "padding": 1},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 32, "use_bn": True, "padding": 1},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 32, "use_bn": True, "padding": 1},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 32, "use_bn": True, "padding": 1},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 64, "use_bn": True, "padding": 1},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 64, "use_bn": True, "padding": 1},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 64, "use_bn": True, "padding": 1},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 64, "use_bn": True, "padding": 1},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 128, "use_bn": True, "padding": 1},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 128, "use_bn": True, "padding": 1}
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
            if use_bn:
                layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU())

            layers.append(nn.MaxPool1d(pool_size))

            # 计算新的宽度
            wd = (wd - kernel_size[0] + 2 * padding + 1) // pool_size[0]
            prev_filters = filters

        layers.append(nn.Flatten())
        layers.append(nn.Linear(prev_filters * wd, 128))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Conv1d_3x_1(nn.Module):
    def __init__(self, input_width, input_channels, output_dim, dropout=0.5):
        super(Conv1d_3x_1, self).__init__()

        self.conv = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(32 * (input_width // 2), 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = nn.ReLU()(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.dropout(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x


class Conv1d_3x_3(nn.Module):
    def __init__(self, input_width, input_channels, output_dim, dropout=0.5):
        super(Conv1d_3x_3, self).__init__()

        # 第一层卷积：input_channels -> 32
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(32)

        # 第二层卷积：32 -> 64
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.bn2 = nn.BatchNorm1d(64)

        # 第三层卷积：64 -> 128
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * (input_width // 8), 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)

        return x


'''
rnns
-----------------------------
'''


class BasicRNN(nn.Module):  # too shit, discard
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


class BasicLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(BasicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            dropout=dropout)
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = tc.zeros(1, x.size(0), self.hidden_dim).to(x.device)  # 初始化隐藏状态
        c0 = tc.zeros(1, x.size(0), self.hidden_dim).to(x.device)  # 初始化单元状态
        out, _ = self.lstm(x.float(), (h0, c0))
        out = out[:, -1, :]  # 只取序列的最后一个时间步的输出
        outputs = self.fc(out)
        return outputs
