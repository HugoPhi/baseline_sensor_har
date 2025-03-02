from plugins.lrkit.executer import Executer
import toml

from data_process import X_train, y_train, X_test, y_test
from clfs import DecisionTreeClf, RandomForestClf, XGBClf, AdaBoostClf, SVClf, LGBMClf, MLPClf


config = toml.load('./hyper.toml')

clf_name = ['DecisionTree', 'RandomForest', 'XGBoost', 'AdaBoost', 'SVM', 'LightGBM', 'MLP']
clf_list = [DecisionTreeClf, RandomForestClf, XGBClf, AdaBoostClf, SVClf, LGBMClf, MLPClf]

clfs = {k: v(**config[k]) for k, v in zip(clf_name, clf_list)}

exc = Executer(X_train, y_train, X_test, y_test,
               clf_dict=clfs,
               log=False,
               log_dir='./log/')

exc.run_all(sort_by='accuracy', ascending=False)
