from sklearn import metrics
from sklearn.model_selection import cross_validate


class Classifier:
    def __init__(self, svm, knn, rf):
        self.svm = svm
        self.knn = knn
        self.rf = rf

    def svm_learn(self, train_X, train_Y):
        self.svm.fit(train_X, train_Y)
        return self.svm

    def svm_predict(self, test_X):
        results = self.svm.predict(test_X)
        return results

    def svm_accuracy(self, test_X, test_Y):
        results = self.svm_predict(test_X)
        return metrics.accuracy_score(test_Y, results)

    def knn_learn(self, train_X, train_Y):
        self.knn.fit(train_X, train_Y)
        return self.knn

    def knn_predict(self, test_X):
        results = self.knn.predict(test_X)
        return results

    def knn_accuracy(self, test_X, test_Y):
        results = self.knn_predict(test_X)
        return metrics.accuracy_score(test_Y, results)

    def rf_learn(self, train_X, train_Y):
        self.rf.fit(train_X, train_Y)
        return self.rf

    def rf_predict(self, test_X):
        results = self.rf.predict(test_X)
        return results

    def rf_accuracy(self, test_X, test_Y):
        results = self.rf_predict(test_X)
        return metrics.accuracy_score(test_Y, results)

    def cross_validate(self, estimator, X, y, cv=5):
        scoring = ['accuracy', 'f1_micro']
        results = cross_validate(estimator=estimator, X=X, y=y, cv=cv, scoring=scoring, return_train_score=True)
        return {"Mean Training Accuracy Score": results['train_accuracy'].mean() * 100,
                "Mean Training f1_macro Score": results['train_f1_micro'].mean(),
                "Mean Validation Accuracy Score": results['test_accuracy'].mean() * 100,
                "Mean Validation f1_macro Score": results['test_f1_micro'].mean()}
