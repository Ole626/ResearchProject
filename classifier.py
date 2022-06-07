from sklearn import svm, metrics


class Classifier:
    def __init__(self):
        self.svm = None

    def svm_learn(self, train_X, train_Y):
        self.svm = svm.SVC(decision_function_shape='ovo')
        self.svm.fit(train_X, train_Y)
        return self.svm

    def svm_predict(self, test_X):
        results = self.svm.predict(test_X)
        return results

    def svm_accuracy(self, test_X, test_Y):
        results = self.svm_predict(test_X)
        return metrics.accuracy_score(test_Y, results)
