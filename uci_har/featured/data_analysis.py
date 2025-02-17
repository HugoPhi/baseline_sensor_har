import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class DataAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        self.info()
        # self.distribute()
        self.statistics()
        self.exceptions()
        self.correlation(vis=False)

    def info(self):
        # 基本信息
        print('\n\n\n\n--------------------------基本信息--------------------------')

        self.df.info()

        print('============================================================')

    def distribute(self):
        for column in self.df.columns:
            plt.figure(figsize=(10, 6))  # 可选：设置图表大小

            # 检查列的数据类型
            if pd.api.types.is_numeric_dtype(self.df[column]):
                # 如果是数值类型，使用distplot或histplot来展示数据分布
                sns.histplot(self.df[column], kde=False)
                plt.title(f'Histogram of {column}')
            elif pd.api.types.is_categorical_dtype(self.df[column]) or self.df[column].nunique() < 10:
                # 如果是分类数据或者唯一值数量小于10（视为分类）
                sns.countplot(x=column, data=self.df)
                plt.title(f'Count Plot of {column}')
            else:
                # 对于其他类型的数据，尝试转换为字符串类型后作为分类数据处理
                self.df[column] = self.df[column].astype(str)
                sns.countplot(x=column, data=self.df)
                plt.title(f'Count Plot of {column} (as string)')

            plt.show()

    def statistics(self):
        print('\n\n\n\n------------------------每列统计信息------------------------')

        print(self.df.describe())

        print('============================================================')

    def exceptions(self):
        # 缺失值
        print('\n\n\n\n------------------------缺失、异常值------------------------')

        print("* 是否空：")
        print(self.df.isnull().sum())
        print("* 是否Nan：")
        print(self.df.isna().sum())

        print('============================================================')

    def correlation(self, vis: bool):
        # 相关性矩阵
        print('\n\n\n\n-------------------------相关性矩阵-------------------------')

        print(self.df.head(5).corr())

        if vis:
            corr = self.df.head(5).corr()
            plt.figure(figsize=(8, 6))  # 设置图表大小
            sns.heatmap(corr, annot=False, cmap='coolwarm')
            plt.title('Correlation Matrix Heatmap')  # 添加标题
            plt.show()  # 显示图形

        print('============================================================')


# 加载训练集
X_train = pd.read_csv(
    "../data/train/X_train.txt",
    sep=r'\s+',
    header=None,
    engine='python')

with open('../data/features.txt', 'r') as file:
    headers = [line.split(' ', 1)[1].strip() for line in file.readlines()]

X_train.columns = headers

RANGE = [headers[x] for x in range(6)]

DataAnalysis(X_train[RANGE])

Y_train = pd.read_csv(
    "../data/train/y_train.txt",
    sep=r'\s+',
    header=None,
    engine='python'
)

DataAnalysis(Y_train)

# 加载测试集
X_test = pd.read_csv(
    "../data/test/X_test.txt",
    sep=r'\s+',
    header=None,
    engine='python')

with open('../data/features.txt', 'r') as file:
    headers = [line.split(' ', 1)[1].strip() for line in file.readlines()]

X_test.columns = headers

RANGE = [headers[x] for x in range(6)]

DataAnalysis(X_test[RANGE])

Y_test = pd.read_csv(
    "../data/test/y_test.txt",
    sep=r'\s+',
    header=None,
    engine='python'
)

DataAnalysis(Y_test)
# np.set_printoptions(suppress=True, precision=5)
# val, train = np.unique(Y_train.to_numpy(), return_counts=True)
# _, test = np.unique(Y_test.to_numpy(), return_counts=True)
# print(np.stack((val, train, test, train / test)))
