import torch as tc
import joblib
from abc import ABC, abstractmethod
import time
import numpy as np


class Clfs(ABC):
    # 1. 快速取用
    # 2. 参数信息获取
    # 3. 同一API

    @abstractmethod
    def __init__(self):
        self.model = None
        self.x = None
        self.y = None
        self.training_time = -1
        self.testing_time = -1

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


class MLClfs(Clfs):
    def __init__(self, model, x, y, param_file=None):
        '''
        x: 训练集
        y: 测试集
        '''

        self.model = model
        self.x = x
        self.y = y
        self.param_file = 'model_filename.pkl'
        self.training_time = -1
        self.testing_time = -1

    def fit(self, load=False):
        if load:
            self.model = joblib.load('model_filename.pkl')

        start = time.time()
        self.model.fit(self.x, self.y.ravel())
        end = time.time()
        self.training_time = end - start

    def predict(self, x_test):
        start = time.time()
        y_pred = self.model.predict(x_test)
        end = time.time()
        self.testing_time = end - start
        return y_pred.squeeze()

    def hyper_info(self):
        return self.model.get_params()


class XGBClfs(Clfs):
    def __init__(self, model, x, y, param_file=None):
        '''
        x: 训练集
        y: 测试集
        '''

        self.model = model
        self.x = x
        self.y = y
        self.param_file = 'model_filename.pkl'
        self.training_time = -1
        self.testing_time = -1

    def fit(self, load=False):
        if load:
            self.model = joblib.load('model_filename.pkl')

        start = time.time()
        self.model.fit(self.x, self.y.ravel() - 1)
        end = time.time()
        self.training_time = end - start

    def predict(self, x_test):
        start = time.time()
        y_pred = self.model.predict(x_test)
        end = time.time()
        self.testing_time = end - start
        return y_pred.squeeze() + 1

    def hyper_info(self):
        return self.model.get_params()


class MLPClf(Clfs):
    def __init__(self, X, y):
        # 将标签重新映射为从 0 开始的连续整数
        y = y - 1

        # 获取数据集参数
        input_size = X.shape[1]
        num_classes = len(np.unique(y))  # 类别数

        # 定义MLP模型

        class MLP(tc.nn.Module):
            def __init__(self, input_size, output_size):
                super(MLP, self).__init__()
                self.fc1 = tc.nn.Linear(input_size, 128)
                self.fc2 = tc.nn.Linear(128, 64)
                self.fc3 = tc.nn.Linear(64, 32)
                self.fc4 = tc.nn.Linear(32, output_size)

            def forward(self, x):
                x = tc.relu(self.fc1(x))
                x = tc.relu(self.fc2(x))
                x = tc.relu(self.fc3(x))
                x = self.fc4(x)  # CrossEntropyLoss会自动处理logits
                return x

        # 初始化模型
        self.model = MLP(input_size, num_classes)

        # 定义损失函数和优化器
        self.criterion = tc.nn.CrossEntropyLoss()
        self.optimizer = tc.optim.Adam(self.model.parameters(), lr=0.001)

        # 转换为Tensor并统一数据类型
        self.X_train_tensor = tc.tensor(X, dtype=tc.float32)
        self.y_train_tensor = tc.tensor(y, dtype=tc.long).reshape(-1)

    def fit(self, load=False):
        # 训练循环
        start = time.time()
        for epoch in range(500):
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train_tensor)
            loss = self.criterion(outputs, self.y_train_tensor)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
        end = time.time()
        self.training_time = end - start

    def predict(self, x_test):
        # 测试阶段
        start = time.time()
        self.model.eval()
        with tc.no_grad():
            X_test_tensor = tc.tensor(x_test, dtype=tc.float32)
            test_outputs = self.model(X_test_tensor)
            y_pred = test_outputs.argmax(dim=1).numpy()

        end = time.time()
        self.testing_time = end - start

        return y_pred.squeeze() + 1

    def hyper_info(self):
        return {
            "hidden_size": "561 -> 128 -> 64 -> 32 -> 6",
            "lr": 0.001,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "epochs": 500
        }
