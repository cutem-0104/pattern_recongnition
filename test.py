from sklearn import datasets
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
images = datasets.load_digits()

# 学習データ
data = images.data

# 教師データ(ラベル)
target = images.target
# テストデータの比率
size = 0.2
# トレーニングデータとテストデータの準備
# random_stateでサンプルのとり方を決める
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, target, test_size=size, random_state=10)
# 別のデータを準備
X_train_1, X_test_1, y_train_1, y_test_1 = cross_validation.train_test_split(data, target, test_size=size, random_state=100)
# データの表示
print([d.shape for d in [X_train, X_test, y_train, y_test]])

# 判別器はSVMを利用
classifier = svm.SVC(C=1.0, kernel='linear')
# 学習
fitted = classifier.fit(X_train, y_train)
# 再代入誤り率の表示
print(u"再代入誤り率:", 1 - fitted.score(X_train, y_train))
# ホールドアウト誤り率の表示
print(u"ホールドアウト誤り率:", 1 - fitted.score(X_test, y_test))

# 学習データと教師データの重複ありの誤り率を表示
print(u"誤り率(重複):", 1 - fitted.score(X_test_1, y_test_1))

# 判別器はSVMを使う。
classifier_rbf = svm.SVC(C=1.0, kernel='rbf')
# 学習
fitted_rbf = classifier_rbf.fit(X_train, y_train)
# 再代入誤り率の表示
print(u"再代入誤り率(rbf):", 1 - fitted_rbf.score(X_train, y_train))
# ホールドアウト誤り率の表示
print(u"ホールドアウト誤り率(rbf):", 1 - fitted_rbf.score(X_test, y_test))
# 学習データと教師データの重複ありの誤り率を表示
print(u"誤り率(rbf 重複):", 1 - fitted_rbf.score(X_test_1, y_test_1))

# 予測
predicted = fitted.predict(X_test)
# 予測結果の表示
print(metrics.confusion_matrix(predicted, y_test))
# 精度表示
print(metrics.accuracy_score(predicted, y_test))