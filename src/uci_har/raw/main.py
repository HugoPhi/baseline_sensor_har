import yaml

from plugins.lrkit.executer import NonValidExecuter

from models.inertial import *

from data_process import X_train, y_train, X_test, y_test  # get dataset


with open('./hyper.yml', 'r') as f:
    clfs = yaml.unsafe_load(f)

exc = NonValidExecuter(X_train, y_train, X_test, y_test,
                       clf_dict=clfs,
                       log=False,
                       log_dir='./log/')

# exc.run_all(sort_by='accuracy', ascending=False, time=True)
mtc, _ = exc.run('gru')
print(mtc.matrix)

exc.format_print()
