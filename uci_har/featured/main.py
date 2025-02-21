import toml
import atexit
from datetime import datetime
import os
import traceback
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from clfs import MLClfs, MLPClf, XGBClfs
from data_process import X_train, y_train, X_test, y_test
from utils import Metrics
from models import MLP


config = toml.load('./hyper.toml')

ml_models = {
    'DecisionTree': DecisionTreeClassifier(**config['DecisionTree']['model']),
    'RandomForest': RandomForestClassifier(**config['RandomForest']['model']),
    'XGBoost': XGBClassifier(**config['XGBoost']['model']),
    'AdaBoost': AdaBoostClassifier(**config['AdaBoost']['model'], base_estimator=DecisionTreeClassifier(max_depth=7)),
    'SVM': SVC(**config['SVM']['model']),
    'LightGBM': LGBMClassifier(**config['LightGBM']['model']),
}

clfs = dict()
for name, model in ml_models.items():
    if name == 'XGBoost':
        clfs[name] = XGBClfs(**config[name]['hyper'], model=model)
    else:
        clfs[name] = MLClfs(**config[name]['hyper'], model=model)

clfs['MLP'] = MLPClf(**config['mlp']['hyper'], model=MLP(input_size=X_train.shape[1], output_size=6))


class Excuter:
    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['acc', 'macro_f1', 'micro_f1', 'avg_recall'],
                 log=False,
                 log_dir='./log/'):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.clf_dict = clf_dict

        self.df = pd.DataFrame(columns=['model'] + metric_list + ['time'])

        # log
        if log:
            self.log_dir = log_dir
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            self.log_path = os.path.join(self.log_dir, f'{datetime.now().strftime("%Y_%m_%d_%H-%M-%S")}/')
            os.mkdir(self.log_path)

            hyper_config = dict()
            for name, clf in clf_dict.items():
                hyper_config[name] = dict()
                hyper_config[name]['hyper'], hyper_config[name]['model'] = clf.hyper_info()

            toml.dump(hyper_config, open(os.path.join(self.log_path, 'hyper.toml'), 'w'))  # 保存超参数和模型参数

            atexit.register(self.save_df)  # 保证退出的时候能保存已经生成的df

    def save_df(self):
        self.df.to_csv(os.path.join(self.log_path, 'result.csv'), index=False)

    def excute(self, name, clf):
        print(f'>> {name}')

        clf.fit(self.X_train, self.y_train)
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict_proba(X_test)
        mtc = Metrics(y_test - 1, y_pred)
        self.df.loc[len(self.df)] = [name, mtc.accuracy(), mtc.macro_f1(), mtc.micro_f1(), mtc.avg_recall(), clf.get_training_time()]

    def run(self, key):
        if key in self.clf_dict.keys():
            self.excute(key, self.clf_dict[key])
        else:
            raise KeyError(f'{key} is not in clf_dict')

    def step(self):
        if len(self.clf_dict) == 0:
            return None

        try:
            name, clf = self.clf_dict.popitem()
            self.excute(name, clf)
            return name, clf
        except Exception as e:
            print(f'Error: {e}')
            traceback.print_exc()

    def run_all(self):
        for name, clf in self.clf_dict.items():
            self.excute(name, clf)

        print(self.df.sort_values('acc', ascending=False))

    def result(self):
        return self.df


exc = Excuter(X_train, y_train, X_test, y_test,
              clf_dict=clfs,
              log=True,
              log_dir='./log/')


exc.run_all()
