import toml
from plugins.excuter import Excuter

from data_process import X_train, y_train, X_test, y_test
from clfs import DecisionTreeClf, RandomForestClf, XGBClf, AdaBoostClf, SVClf, LGBMClf, MLPClf


config = toml.load('./hyper.toml')

clfs = {
    'DecisionTree': DecisionTreeClf(**config['DecisionTree']),
    'RandomForest': RandomForestClf(**config['RandomForest']),
    'XGBoost': XGBClf(**config['XGBoost']),
    'AdaBoost': AdaBoostClf(**config['AdaBoost']),
    'SVM': SVClf(**config['SVM']),
    'LightGBM': LGBMClf(**config['LightGBM']),
    'MLP': MLPClf(**config['MLP'])
}


exc = Excuter(X_train, y_train, X_test, y_test,
              clf_dict=clfs,
              log=True,
              log_dir='./log/')
exc.run_all(sort_by='accuracy', ascending=False)
