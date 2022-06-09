from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier


class Classifier:
    def __init__(self):
        self.svm = None
        self.knn = None

    def svm_learn(self, train_X, train_Y):
        self.svm = svm.SVC(kernel='linear', C=100)
        self.svm.fit(train_X, train_Y)
        return self.svm

    def svm_predict(self, test_X):
        results = self.svm.predict(test_X)
        return results

    def svm_accuracy(self, test_X, test_Y):
        results = self.svm_predict(test_X)
        return metrics.accuracy_score(test_Y, results)

    def knn_learn(self, train_X, train_Y):
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(train_X, train_Y)
        return self.knn

    def knn_predict(self, test_X):
        results = self.knn.predict(test_X)
        return results

    def knn_accuracy(self, test_X, test_Y):
        results = self.knn_predict(test_X)
        return metrics.accuracy_score(test_Y, results)
