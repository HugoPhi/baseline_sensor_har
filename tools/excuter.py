import atexit
import toml
from datetime import datetime
import os
import traceback
import pandas as pd
from .metric import Metrics


class Excuter:
    '''
    Clf的执行器
    ==========
      - 快捷管理训练，测试，日志全过程，并灵活调试Classifier数组里面的各个模型。
      - 开启Log，支持中途运行出错，结果不丢失。
      - 在使用的时候需要根据metric_list重写excute(self)方法。

    Parameters
    ----------
    X_train : np.ndarray
        训练集的X。
    y_train : np.ndarray
        训练集的y。
    X_test : np.ndarray
        测试集的X。
    y_test : np.ndarray
        测试集的y。
    clf_dict : dict
        Clf字典。包含多个实验的{name : Clf}
    metric_list : list
        测评指标列表。在两端分别加上name和time之后，作为结果表格的表头。
    log : bool
        是否开启日志。开启之后会将过程参数写入到对应文件夹的hyper.toml，将结果写入到同一文件夹的result.csv。
    log_dir : str
        存放日志的文件夹。日志会被放到一个日期为名字的子文件夹里面。
    '''

    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['acc', 'macro_f1', 'micro_f1', 'avg_recall'],
                 log=False,
                 log_dir='./log/'):
        '''
        初始化。

        Parameters
        ----------
        X_train : np.ndarray
            训练集的X。
        y_train : np.ndarray
            训练集的y。
        X_test : np.ndarray
            测试集的X。
        y_test : np.ndarray
            测试集的y。
        clf_dict : dict
            Clf字典。包含多个实验的{name : Clf}
        metric_list : list
            测评指标列表。在两端分别加上name和time之后，作为结果表格的表头。
        log : bool
            是否开启日志。开启之后会将过程参数写入到对应文件夹的hyper.toml，将结果写入到同一文件夹的result.csv。
        log_dir : str
            存放日志的文件夹。日志会被放到一个日期为名字的子文件夹里面。
        '''

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
                hyper_config[name]['hyper'], hyper_config[name]['model'] = clf.get_params()

            toml.dump(hyper_config, open(os.path.join(self.log_path, 'hyper.toml'), 'w'))  # 保存超参数和模型参数

            atexit.register(self.save_df)  # 保证退出的时候能保存已经生成的df

    def save_df(self):
        '''
        保存df到日志
        '''
        self.df.to_csv(os.path.join(self.log_path, 'result.csv'), index=False)

    def excute(self, name, clf):
        '''
        执行实验。

        Notes
        -----
          - 需要根据metric_list重新实现的部分。这部分你需要定义一个实验需要做的事情。
          - 这里必须要做的就是把self.df的更新方式改变，把对应的表头下填入一次实验得到的正确数据。

        Parameters
        ----------
        name : str
            实验的名字。
        clf : lib.clfs.Clfs
            实验获取的分类器，继承自接口lib.clfs.Clfs。

        Examples
        --------

        ```python
        print(f'>> {name}')

        clf.fit(self.X_train, self.y_train)
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict_proba(self.X_test)

        mtc = Metrics(self.y_test, y_pred)

        self.df.loc[len(self.df)] = [name, mtc.accuracy(), mtc.macro_f1(), mtc.micro_f1(), mtc.avg_recall(), clf.get_training_time()]
        ```
        '''
        print(f'>> {name}')

        clf.fit(self.X_train, self.y_train)
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict_proba(self.X_test)

        mtc = Metrics(self.y_test, y_pred)

        self.df.loc[len(self.df)] = [name, mtc.accuracy(), mtc.macro_f1(), mtc.micro_f1(), mtc.avg_recall(), clf.get_training_time()]

    def run(self, key):
        '''
        运行单个实验。不会消耗clf_dict。

        Parameters
        ----------
        key : str
            实验的名字。
        '''
        if key in self.clf_dict.keys():
            self.excute(key, self.clf_dict[key])
        else:
            raise KeyError(f'{key} is not in clf_dict')

    def step(self):
        '''
        迭代运行实验。采用迭代器模式。会逐个消耗实验，直到clf_dict为空。过程中会返回对应的名字和Clf对象，如果是最后一个，返回None。

        Returns
        -------
        name : str
            实验的名字。
        clf : lib.clfs.Clfs
            实验获取的分类器，继承自接口lib.clfs.Clfs。
        '''
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
        '''
        运行所有实验。
        '''
        for name, clf in self.clf_dict.items():
            self.excute(name, clf)

        print(self.df.sort_values('acc', ascending=False))

    def result(self):
        '''
        返回实验结果对应的表格。

        Returns
        -------
        self.pd : pd.DataFrame
        '''
        return self.df
