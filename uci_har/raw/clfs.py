import torch
from torch import nn
from abc import ABC, abstractmethod
import time


class Clfs(ABC):
    # 1. 快速取用
    # 2. 参数信息获取
    # 3. 同一API

    @abstractmethod
    def predict(self):
        '''
        !!! 在这里要统计预测时间
        '''
        pass

    @abstractmethod
    def fit(self, load=False):
        '''
        !!! 在这里要统计测试时间
        '''
        pass

    @abstractmethod
    def hyper_info(self) -> dict:
        pass

    @abstractmethod
    def get_training_time(self):
        pass

    @abstractmethod
    def get_testing_time(self):
        pass


class MLPClf(Clfs):
    '''
    Example:

    ```python
    config = {
        "input_dim": X_train.shape[1] * X_train.shape[2],
        "hidden_dims": [256, 128, 64, 32],
        "output_dim": 6,
        "dropout": 0.2,
        "lr": 0.001,
        "epochs": 100,
        "batch_size": 32
    }

    mlp_clf = MLPClf(X_train, y_train, config)
    mlp_clf.fit()

    # 预测
    y_pred = mlp_clf.predict(X_test)
    acc = (y_pred == (y_test - 1)).float().mean().item()
    print(acc)
    ```

    '''

    def __init__(self, X, y, config, pre_trained="./model.m5"):

        super().__init__()
        self.config = config
        self.input_dim = config["input_dim"]  # 输入层维度
        self.hidden_dims = config["hidden_dims"]  # 隐藏层维度列表
        self.output_dim = config["output_dim"]  # 输出层维度
        self.dropout = config["dropout"]  # Dropout 比例
        self.lr = config["lr"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.pre_trained = pre_trained

        self.X = X
        self.y = y

        self.training_time = -1
        self.testing_time = -1

        # 构建模型
        layers = []
        prev_dim = self.input_dim
        for hdim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.model = nn.Sequential(*layers)

        self.criterion = nn.CrossEntropyLoss()  # 损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  # 优化器

    def fit(self, load=False):
        if not load:
            X_train = self.X.reshape(self.X.shape[0], -1)
            y_train = self.y
            dataset = torch.utils.data.TensorDataset(X_train, y_train)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            start_time = time.time()
            for epoch in range(self.epochs):
                for i, (x, y) in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    outputs = self.model(x.float())
                    loss = self.criterion(outputs, torch.argmax(y, dim=1))
                    loss.backward()
                    self.optimizer.step()

                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}')
            self.training_time = time.time() - start_time
        else:
            print("Loading pre-trained model...")
            self.model.load_state_dict(torch.load(self.pre_trained))
            print("Pre-trained model loaded.")

    def predict(self, X_test):
        start_time = time.time()
        X_test = X_test.reshape(X_test.shape[0], -1)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(X_test.float())
            _, predicted = torch.max(outputs.data, 1)
        self.testing_time = time.time() - start_time
        return predicted

    def hyper_info(self) -> dict:
        self.config.pop("input_dim")
        self.config.pop("hidden_dims")
        self.config.pop("output_dim")
        self.config["arch"] = self.model.__str__()
        return self.config

    def get_training_time(self):
        return self.training_time

    def get_testing_time(self):
        return self.testing_time


class Conv2d1cClf(Clfs):
    '''
    Example:

    ```python
    config = {
        "output_dim": 6,
        "dropout": 0.5,
        "lr": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "layers_config": [
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 32, "use_bn": True},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 64, "use_bn": True},
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 128, "use_bn": False}
        ]
    }

    cnn_clf = CnnClf(X_train, y_train, config)
    cnn_clf.fit()

    # 预测
    y_pred = cnn_clf.predict(X_test)
    acc = (y_pred == (y_test - 1)).float().mean().item()
    print(acc)
    ```

    '''

    def __init__(self, X, y, config, pre_trained="./model_cnn3.m5"):
        super().__init__()
        self.config = config
        self.input_channels = 1
        self.input_height = X.shape[1]
        self.input_width = X.shape[2]
        self.output_dim = config["output_dim"]
        self.layers_config = config["layers_config"]
        self.dropout = config["dropout"]
        self.lr = config["lr"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.pre_trained = pre_trained
        self.X = X
        self.y = y

        self.training_time = -1
        self.testing_time = -1

        # 构建模型
        layers = []
        wd, he = self.input_width, self.input_height
        prev_filters = self.input_channels
        for layer_config in self.layers_config:
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

            layers.append(nn.Dropout(self.dropout))

            # W/H here is width/height, they are the same
            # Conv      : W <- (W + 2P - K) // S + 1
            # MaxPooling: W <- W // 2
            he = (he - kernel_size[0] + 2 * padding + 1) // pool_size[0]
            wd = (wd - kernel_size[1] + 2 * padding + 1) // pool_size[1]
            prev_filters = filters

        layers.append(nn.Flatten())
        layers.append(nn.Linear(prev_filters * wd * he, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, self.output_dim))

        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, load=False):
        if not load:
            X_train = self.X.reshape(self.X.shape[0], self.input_channels, self.input_height, self.input_width)
            y_train = self.y
            dataset = torch.utils.data.TensorDataset(X_train, y_train)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            start_time = time.time()
            for epoch in range(self.epochs):
                for i, (x, y) in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    outputs = self.model(x.float())
                    loss = self.criterion(outputs, torch.argmax(y, dim=1))
                    loss.backward()
                    self.optimizer.step()

                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}')
            self.training_time = time.time() - start_time
        else:
            print("Loading pre-trained model...")
            self.model.load_state_dict(torch.load(self.pre_trained))
            print("Pre-trained model loaded.")

    def predict(self, X_test):
        start_time = time.time()
        X_test = X_test.reshape(X_test.shape[0], self.input_channels, self.input_height, self.input_width)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(X_test.float())
            _, predicted = torch.max(outputs.data, 1)
        self.testing_time = time.time() - start_time
        return predicted

    def hyper_info(self) -> dict:
        self.config.pop("layers_config")
        self.config["arch"] = self.model.__str__()
        return self.config

    def get_training_time(self):
        return self.training_time

    def get_testing_time(self):
        return self.testing_time


class Conv1dxcClf(Clfs):
    '''
    Example:

    ```python
    config = {
        "output_dim": 6,
        "dropout": 0.5,
        "lr": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "layers_config": [
            {"kernel_size": (3,), "pool_size": (2,), "filters": 32, "use_bn": True},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 64, "use_bn": True},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 128, "use_bn": False}
        ]
    }

    cnn_clf = Conv1d9cClf(X_train, y_train, config)
    cnn_clf.fit()

    # 预测
    y_pred = cnn_clf.predict(X_test)
    acc = (y_pred == (y_test - 1)).float().mean().item()
    print(acc)
    ```

    '''

    def __init__(self, X, y, config, pre_trained="./model_cnn1d.m5"):
        super().__init__()
        self.config = config
        self.input_channels = 9  # 使用9通道
        self.input_height = X.shape[1]  # 高度为9
        self.input_width = X.shape[2]  # 宽度为128
        self.output_dim = config["output_dim"]
        self.layers_config = config["layers_config"]
        self.dropout = config["dropout"]
        self.lr = config["lr"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.pre_trained = pre_trained
        self.X = X
        self.y = y

        self.training_time = -1
        self.testing_time = -1

        # 构建模型
        layers = []
        wd = self.input_width
        prev_filters = self.input_channels
        for layer_config in self.layers_config:
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

            layers.append(nn.Dropout(self.dropout))

            # 计算新的宽度
            wd = (wd - kernel_size[0] + 2 * padding + 1) // pool_size[0]
            prev_filters = filters

        layers.append(nn.Flatten())
        layers.append(nn.Linear(prev_filters * wd, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, self.output_dim))

        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, load=False):
        if not load:
            X_train = self.X.reshape(self.X.shape[0], self.input_channels, self.input_width)
            y_train = self.y
            dataset = torch.utils.data.TensorDataset(X_train, y_train)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            start_time = time.time()
            for epoch in range(self.epochs):
                for i, (x, y) in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    outputs = self.model(x.float())
                    loss = self.criterion(outputs, torch.argmax(y, dim=1))
                    loss.backward()
                    self.optimizer.step()

                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}')
            self.training_time = time.time() - start_time
        else:
            print("Loading pre-trained model...")
            self.model.load_state_dict(torch.load(self.pre_trained))
            print("Pre-trained model loaded.")

    def predict(self, X_test):
        start_time = time.time()
        X_test = X_test.reshape(X_test.shape[0], self.input_channels, self.input_width)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(X_test.float())
            _, predicted = torch.max(outputs.data, 1)
        self.testing_time = time.time() - start_time
        return predicted

    def hyper_info(self) -> dict:
        self.config.pop("layers_config")
        self.config["arch"] = self.model.__str__()
        return self.config

    def get_training_time(self):
        return self.training_time

    def get_testing_time(self):
        return self.testing_time
