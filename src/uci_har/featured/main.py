import yaml
from plugins.lrkit.executer import NonValidExecuter

from data_process import X_train, y_train, X_test, y_test

from models.feature import *


with open('./hyper.yml') as f:
    clfs = yaml.unsafe_load(f)['models']

exc = NonValidExecuter(X_train, y_train, X_test, y_test,
                       clf_dict=clfs,
                       log=True,
                       log_dir='./log/')

exc.run_all(sort_by='accuracy', ascending=False, time=True)
