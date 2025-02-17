import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


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


PATH = '../data/train/Inertial Signals/'
PREFIXS = [
    'body_acc_x_',
    'body_acc_y_',
    'body_acc_z_',
    'body_gyro_x_',
    'body_gyro_y_',
    'body_gyro_z_',
    'total_acc_x_',
    'total_acc_y_',
    'total_acc_z_',
]


X_train = []
for prefix in PREFIXS:
    X_train.append(pd.read_csv(PATH + prefix + 'train.txt', header=None, delim_whitespace=True).to_numpy())
X_train = np.transpose(np.array(X_train), (1, 0, 2))

y_train = pd.read_csv('../data/train/y_train.txt', header=None).to_numpy()
print(X_train.shape)
print(y_train.shape)
