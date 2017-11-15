import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn import svm, datasets, cross_validation
from sklearn.cross_validation import StratifiedKFold
digits = datasets.load_digits()

#学習データ
X_digits = digits.data
#教師データ
y_digits = digits.target

# stratifiedKFold法
skf = StratifiedKFold(y_digits, 5)
for train, test in skf:
    print("training data indices", train)
    print("test data indices", test)
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_digits[train], y_digits[train])
    predict = clf.predict(X_digits[test])
    print("prediction", predict)
    print("ground truth", y_digits[test])
    print("accuracy", (predict == y_digits[test]).sum() / float(predict.size))
    print("----")


C_list = np.logspace(-8, 2, 11) # C
score = np.zeros((len(C_list),3))
tmp_train, tmp_test = list(), list()
# score_train, score_test = list(), list()
i = 0
for C in C_list:
    svc = svm.SVC(C=C, kernel='rbf', gamma=0.001)
    for train, test in skf:
        svc.fit(X_digits[train], y_digits[train])
        tmp_train.append(svc.score(X_digits[train],y_digits[train]))
        tmp_test.append(svc.score(X_digits[test],y_digits[test]))
        score[i,0] = C
        score[i,1] = sum(tmp_train) / len(tmp_train)
        score[i,2] = sum(tmp_test) / len(tmp_test)
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