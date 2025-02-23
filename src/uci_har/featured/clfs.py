import time
import torch as tc
from plugins.clfs import Clfs

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from models import MLP
from data_process import X_train


class SklearnClfs(Clfs):
    def __init__(self):
        super(SklearnClfs, self).__init__()

        self.model = None
        self.training_time = -1
        self.testing_time = -1

    def fit(self, X_train, y_train, load=False):
        start = time.time()
        self.model.fit(X_train, y_train.ravel())
        end = time.time()
        self.training_time = end - start

    def predict(self, x_test):
        start = time.time()
        y_pred = self.model.predict(x_test)
        end = time.time()
        self.testing_time = end - start
        return y_pred.squeeze()

    def predict_proba(self, x_test):
        start = time.time()
        y_pred = self.model.predict_proba(x_test)
        end = time.time()
        self.testing_time = end - start
        return y_pred

    def get_training_time(self):
        return self.training_time

    def get_testing_time(self):
        return self.testing_time


class DecisionTreeClf(SklearnClfs):
    def __init__(self, **kwargs):
        super(DecisionTreeClf, self).__init__()

        self.model = DecisionTreeClassifier(**kwargs)


class RandomForestClf(SklearnClfs):
    def __init__(self, **kwargs):
        super(RandomForestClf, self).__init__()

        self.model = RandomForestClassifier(**kwargs)


class XGBClf(SklearnClfs):
    def __init__(self, **kwargs):
        super(XGBClf, self).__init__()

        self.model = XGBClassifier(**kwargs)


class AdaBoostClf(SklearnClfs):
    def __init__(self, **kwargs):
        super(AdaBoostClf, self).__init__()

        self.model = AdaBoostClassifier(**kwargs, estimator=DecisionTreeClassifier(max_depth=7))


class LGBMClf(SklearnClfs):
    def __init__(self, **kwargs):
        super(LGBMClf, self).__init__()

        self.model = LGBMClassifier(**kwargs)


class SVClf(SklearnClfs):
    def __init__(self, **kwargs):
        super(SVClf, self).__init__()

        self.model = SVC(**kwargs)


class MLPClf(Clfs):
    def __init__(self, epochs, lr):
        super(MLPClf, self).__init__()

        self.epochs = epochs
        self.lr = lr
        self.model = MLP(input_size=X_train.shape[1], output_size=6)

        # 定义损失函数和优化器
        self.criterion = tc.nn.CrossEntropyLoss()
        self.optimizer = tc.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X_train, y_train, load=False):
        # 训练循环
        X_train_tensor = tc.tensor(X_train, dtype=tc.float32)
        y_train_tensor = tc.tensor(y_train, dtype=tc.long).reshape(-1)
        start = time.time()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
        end = time.time()
        self.training_time = end - start

    def predict_proba(self, x_test):
        # 测试阶段
        start = time.time()
        self.model.eval()
        with tc.no_grad():
            X_test_tensor = tc.tensor(x_test, dtype=tc.float32)
            test_outputs = self.model(X_test_tensor)
            test_outputs = tc.nn.functional.softmax(test_outputs, dim=1)
        end = time.time()
        self.testing_time = end - start
        return test_outputs.numpy()

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
        return y_pred.squeeze()  # class from [0, 1, 2, 3, 4, 5]

    def get_training_time(self):
        return self.training_time

    def get_testing_time(self):
        return self.testing_time
