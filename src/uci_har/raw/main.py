import yaml
from data_process import X_train, y_train, X_test, y_test  # get dataset

from plugins.lrkit.executer import Executer
from clfs import *

with open('./hyper.yml', 'r') as f:
    clfs = yaml.unsafe_load(f)

exc = Executer(X_train, y_train, X_test, y_test,
               clf_dict=clfs,
               log=False,
               log_dir='./log/')

exc.run_all(sort_by='accuracy', ascending=False)
