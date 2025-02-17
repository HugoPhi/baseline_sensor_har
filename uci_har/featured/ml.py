import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from clfs import MLClfs, MLPClf, XGBClfs

LOG_IX = 3
SEED = 42
X_train = pd.read_csv(
    "../data/train/X_train.txt",
    sep=r'\s+',
    header=None,
    engine='python').to_numpy()

y_train = pd.read_csv(
    "../data/train/y_train.txt",
    sep=r'\s+',
    header=None,
    engine='python').to_numpy()

X_test = pd.read_csv(
    "../data/test/X_test.txt",
    sep=r'\s+',
    header=None,
    engine='python').to_numpy()

y_test = pd.read_csv(
    "../data/test/y_test.txt",
    sep=r'\s+',
    header=None,
    engine='python').to_numpy()

shuffle = np.random.permutation(len(X_train))
print(f"train: {len(X_train)}, test: {len(X_test)}")
X_train, y_train = X_train[shuffle], y_train[shuffle]


MODELS2 = {
    "DecisionTree": DecisionTreeClassifier(
        criterion='gini',     # 分裂质量的评价标准
        max_depth=None,       # 树的最大深度
        min_samples_split=2,  # 内部节点再划分所需最小样本数
        min_samples_leaf=3,   # 叶节点所需的最小样本数
        random_state=SEED     # 控制随机性以便结果可复现
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,     # 树的数量
        criterion='gini',     # 分裂质量的评价标准
        max_depth=None,       # 树的最大深度
        min_samples_split=4,  # 内部节点再划分所需最小样本数
        min_samples_leaf=1,   # 叶节点所需的最小样本数
        bootstrap=True,       # 是否进行bootstrap采样
        random_state=SEED     # 控制随机性以便结果可复现
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,             # 树的数量
        max_depth=5,                  # 树的最大深度
        learning_rate=0.1,            # 学习率
        subsample=1,                  # 训练每棵树时使用的样本比例
        colsample_bytree=1,           # 建树的特征比例
        objective='binary:logistic',  # 损失函数
        eval_metric='logloss',        # 评价指标
        random_state=SEED             # 控制随机性以便结果可复现
    ),

    # "AdaBoost": AdaBoostClassifier(
    #     base_estimator=DecisionTreeClassifier(max_depth=7),   # 弱学习器
    #     n_estimators=100,    # 弱学习器的数量
    #     learning_rate=1.,   # 对每个弱学习器的贡献程度
    #     random_state=SEED   # 控制随机性以便结果可复现
    # ),

    "SVM": SVC(
        C=0.4,              # 正则化参数
        kernel='rbf',       # 核函数类型
        degree=3,           # 多项式核函数的次数（'poly'时有效）
        gamma='scale',      # 核系数（对于'rbd', 'poly', 'sigmoid'有效）
        probability=False,  # 是否启用概率估计
        random_state=SEED   # 控制随机性以便结果可复现（仅对某些选项有效）
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=200,        # 树的数量
        max_depth=-1,            # 树的最大深度（-1表示没有限制）
        learning_rate=0.1,       # 学习率
        num_leaves=30,           # 一棵树上的叶子数
        subsample=1.0,           # 训练每棵树时使用的样本比例
        colsample_bytree=1.0,    # 建树的特征比例
        objective='multiclass',  # 目标函数
        metric='multi_logloss',  # 评价指标
        random_state=SEED        # 控制随机性以便结果可复现
    )
}

LOG = f'./log/result{LOG_IX}.log'
if os.path.exists(LOG):
    print(f"File {LOG} already exists.")
    exit(0)
else:
    if not os.path.exists('./log'):
        os.makedirs('./log')
    open(LOG, 'a').close()


clf_list = dict()
model_res = dict()
models = MODELS2
for model, v in models.items():
    if model == "XGBoost":
        clf_list[model] = XGBClfs(v, X_train, y_train)
    else:
        clf_list[model] = MLClfs(v, X_train, y_train)

clf_list["MLP"] = MLPClf(X_train, y_train)

for name in clf_list.keys():
    print(f">> {name}: ", end="")
    clf_list[name].fit()
    print(f"{clf_list[name].training_time:.4f} s")

    y_pred = clf_list[name].predict(X_test)

    with open(LOG, 'a') as f:
        f.write(f">> {name}: {clf_list[name].training_time:.4f} s\n")
        for k, v in clf_list[name].hyper_info().items():
            f.write(f" * {k}: {v}\n")
        f.write("\n")

    acc = (y_pred == y_test.reshape(-1)).mean()
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    micro_f1 = f1_score(y_test, y_pred, average="micro")
    model_res[name] = {"acc": acc, "macro_f1": macro_f1, "micro_f1": micro_f1}
    if False:
        print(classification_report(y_test, y_pred))

# 绘制表格
with open(LOG, 'a') as f:
    f.write(f"\n\n\n{'Model':<20} | {'Accuracy':>10} | {'Macro F1':>10} | {'Micro F1':>10}\n")
    f.write("-" * 60 + "\n")

    for k, v in model_res.items():
        f.write(f"{k:<20} | {v['acc']:>10.5f} | {v['macro_f1']:>10.5f} | {v['micro_f1']:>10.5f}\n")
