import numpy as np


class Metrics:
    '''
    # Example:
    ```python
    y_true = np.array([0, 1, 2])         # 真实标签
    y_pred = np.array([[0.7, 0.1, 0.2],  # 对应样本的预测概率，一行为一个样本
                       [0.3, 0.3, 0.4],
                       [0.2, 0.1, 0.7]])
    ```
    尤其是二分类，一定要做成两类：
    ```python
    y_true = np.array([0, 1])       # 真实标签
    y_pred = np.array([[0.9, 0.1],  # 对应样本的预测概率，一行为一个样本
                       [0.3, 0.7],
                       [0.2, 0.8]]
    ```
    '''

    def __init__(self, y, y_pred):
        self.y = y
        self.y_pred = y_pred
        uni = np.unique(self.y)
        if uni[0] != 0:
            raise ValueError("y must start from 0")

        self.classes = uni.shape[0]  # get classes num

        self.matrix = np.zeros((self.classes, self.classes))  # get confusion matrix
        for i, j in zip(y, np.argmax(y_pred, axis=1)):
            self.matrix[i, j] += 1

    def precision(self):
        return np.diag(self.matrix) / self.matrix.sum(axis=0)

    def recall(self):
        return np.diag(self.matrix) / self.matrix.sum(axis=1)

    def f1(self):
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    def accuracy(self):
        return np.diag(self.matrix).sum() / self.matrix.sum()

    def roc(self):
        '''
        we use 'ovr' here, returns auc of each class
        '''

        def calculate_tpr_fpr(y_true, y_pred):
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            return tpr, fpr

        rocs = []
        for class_idx in range(self.classes):
            tprs = []
            fprs = []
            thresholds = self.y_pred[:, class_idx].reshape(-1)
            for threshold in np.sort(thresholds)[::-1]:
                idx_pred = (self.y_pred[:, class_idx] >= threshold).astype(int)  # '=' here is important
                idx_true = (self.y == class_idx).astype(int).reshape(-1)
                tpr, fpr = calculate_tpr_fpr(idx_true, idx_pred)
                tprs.append(tpr)
                fprs.append(fpr)

            rocs.append((tprs, fprs))

        return rocs

    def auc(self):
        '''
        we use 'ovr' here, returns auc of each class
        '''
        rocs = self.roc()
        aucs = []
        for (tprs, fprs) in rocs:
            auc = 0
            for i in range(1, len(fprs)):
                auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2
            aucs.append(auc)

        return np.array(aucs)

    def ap(self):
        '''
        we use 'ovr' here, returns auc of each class
        '''

        def calculate_prec_rec(y_true, y_pred):
            tp = np.sum((y_pred == 1) & (y_true == 1))
            # tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (fp + fn) > 0 else 0
            return prec, rec

        aps = []
        for class_idx in range(self.classes):
            precs = []
            recs = []
            thresholds = self.y_pred[:, class_idx].reshape(-1)
            for threshold in np.sort(thresholds)[::-1]:
                idx_pred = (self.y_pred[:, class_idx] >= threshold).astype(int)  # '=' here is important
                idx_true = (self.y == class_idx).astype(int).reshape(-1)
                prec, rec = calculate_prec_rec(idx_true, idx_pred)
                precs.append(prec)
                recs.append(rec)

            ap = 0
            for i in range(1, len(recs)):
                ap += (recs[i] - recs[i - 1]) * (precs[i] + precs[i - 1]) / 2
            aps.append(ap)

        return aps

    def avg_ap(self):
        return self.ap().mean()

    def avg_pre(self):
        return self.precision().mean()

    def avg_recall(self):
        return self.recall().mean()

    def avg_auc(self):
        return self.auc().mean()

    def macro_f1(self):
        return self.f1().mean()

    def micro_f1(self):
        return 2 * self.precision().mean() * self.recall().mean() / (self.precision().mean() + self.recall().mean())

    def confusion_matrix(self):
        return self.matrix

    def __repr__(self) -> str:
        table = ' ' * 6
        print(f'        {table}Precision{table}Recall{table}  F1')
        for i in range(len(self.precision())):
            print(f'Class {i} {table}{self.precision()[i]:.6f} {table}{self.recall()[i]:.6f}{table}{self.f1()[i]:.6f}')
        print()
        print(f'Accuracy      {self.accuracy():.6f}')
        print(f'Macro avg     {self.macro_avg():.6f}')
        print(f'Micro avg     {self.micro_avg():.6f}')

        return ''
