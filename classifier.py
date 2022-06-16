from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.base import clone
import numpy as np


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

    def cross_validate_over_subjects(self, estimator, X, y, cv=4):
        scoring = ['accuracy', 'f1_micro']
        results = cross_validate(estimator=estimator, X=X, y=y, cv=cv, scoring=scoring, return_train_score=True,
                                 return_estimator=True)

        return {"Mean Training Accuracy Score": results['train_accuracy'].mean() * 100,
                "Mean Training f1_macro Score": results['train_f1_micro'].mean(),
                "Mean Validation Accuracy Score": results['test_accuracy'].mean() * 100,
                "Mean Validation f1_macro Score": results['test_f1_micro'].mean()}, results['estimator']

    def cross_validate_over_activities(self, estimator, X_per_subject, y_per_subject, cv=4):
        estimators = []
        val_accs = []
        f1 = []
        for i in range(0, cv):
            X_train = []
            y_train = []
            X_val = []
            y_val = []
            cloned_est = clone(estimator)

            for X, y in zip(X_per_subject, y_per_subject):
                for X_act, y_act in zip(X, y):
                    X_val += X_act[int(i * (len(X_act) / cv)): int((i+1) * (len(X_act) / cv))]
                    y_val += y_act[int(i * (len(X_act) / cv)): int((i+1) * (len(X_act) / cv))]
                    X_train += [X_act[j] for j in range(0, len(X_act))
                                if j < i * (len(X_act) / cv) or j >= (i+1) * (len(X_act) / cv)]
                    y_train += [y_act[j] for j in range(0, len(X_act))
                                if j < i * (len(X_act) / cv) or j >= (i+1) * (len(X_act) / cv)]

            cloned_est.fit(X_train, y_train)
            estimators.append(cloned_est)
            val_accs.append(metrics.accuracy_score(y_val, cloned_est.predict(X_val)))
            f1.append(metrics.f1_score(y_val, cloned_est.predict(X_val), average='micro'))

        return {"Mean Validation Accuracy Score": np.asarray(val_accs).mean() * 100,
                "Mean Validation f1_macro Score": np.asarray(f1).mean()}, estimators


