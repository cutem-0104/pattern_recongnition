import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn import svm, datasets, cross_validation

digits = datasets.load_digits()

#学習データ
X_digits = digits.data
#教師データ
y_digits = digits.target

#kFold法
np.random.seed(0) # 乱数のシード設定
indices = np.random.permutation(len(X_digits))
print(indices)# [1081 1707  927 ..., 1653  559  684]
X_digits = X_digits[indices] # データの順序をランダムに並び替え
y_digits = y_digits[indices]
n_fold = 4 # 交差検定の回数
# n個の標本データをn_folds個のバッチに分割　1バッチがテスト用、残りをトレーニング用に使用
k_fold = cross_validation.KFold(n=len(X_digits),n_folds = n_fold)

# start～stop区間をnum等分した対数データを生成する
C_list = np.logspace(-8, 2, 11)
#[  1.00000000e-08   1.00000000e-07   1.00000000e-06   1.00000000e-05
#   1.00000000e-04   1.00000000e-03   1.00000000e-02   1.00000000e-01
#   1.00000000e+00   1.00000000e+01   1.00000000e+02]

# 0の配列を作る　C_listのサイズと3
score = np.zeros((len(C_list),3))
#[[ 0.  0.  0.]
# [ 0.  0.  0.]
# [ 0.  0.  0.]
# [ 0.  0.  0.]
# [ 0.  0.  0.]
# [ 0.  0.  0.]
# [ 0.  0.  0.]
# [ 0.  0.  0.]
# [ 0.  0.  0.]
# [ 0.  0.  0.]
# [ 0.  0.  0.]]

tmp_train, tmp_test = list(), list()
# score_train, score_test = list(), list()
i = 0
for C in C_list:
    svc = svm.SVC(C=C, kernel='rbf', gamma=0.001)
    for train, test in k_fold:
        svc.fit(X_digits[train], y_digits[train])
        tmp_train.append(svc.score(X_digits[train],y_digits[train]))
        tmp_test.append(svc.score(X_digits[test],y_digits[test]))
        score[i,0] = C
        score[i,1] = sum(tmp_train) / len(tmp_train)
        score[i,2] = sum(tmp_test) / len(tmp_test)
        # 要素を削除する [:]で先頭の文字から最後の文字までが抽出される
        # [(開始インデックス):(終了インデックス)]
        del tmp_train[:]
        del tmp_test[:]
    i = i + 1

#グラフ化
xmin, xmax = score[:,0].min(), score[:,0].max()
ymin, ymax = score[:,1:2].min()-0.1, score[:,1:2].max()+0.1
plt.semilogx(score[:,0], score[:,1], c = "r", label = "train")
plt.semilogx(score[:,0], score[:,2], c = "b", label = "test")
plt.axis([xmin,xmax,ymin,ymax])
plt.legend(loc='upper left')
plt.xlabel('C')
plt.ylabel('score')
plt.show()