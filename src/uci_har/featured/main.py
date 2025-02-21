import toml
from lib.excuter import Excuter

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from data_process import X_train, y_train, X_test, y_test
from clfs import MLClfs, MLPClf, XGBClfs
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

clfs['MLP'] = MLPClf(**config['MLP']['hyper'], model=MLP(input_size=X_train.shape[1], output_size=6))


exc = Excuter(X_train, y_train, X_test, y_test,
              clf_dict=clfs,
              log=True,
              log_dir='./log/')
exc.run_all()
