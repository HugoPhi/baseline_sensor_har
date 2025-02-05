import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('./UCI HAR Dataset/features.txt', 'r') as file:
    headers = [line.split(' ', 1)[1].strip() for line in file.readlines()]

X_train = pd.read_csv(
    "./UCI HAR Dataset/train/X_train.txt",
    sep=r'\s+',
    header=None,
    engine='python')

X_train.columns = headers
# print(X_train.head(5))
# X_train.info()
# print(X_train[[headers[x] for x in range(6)]].head(5).corr())
# print(X_train[[headers[x] for x in range(6)]].describe())

corr = X_train[[headers[x] for x in range(6)]].head(5).corr()
plt.figure(figsize=(8, 6))  # 设置图表大小
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')  # 添加标题
plt.show()  # 显示图形
