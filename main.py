import sklearn.metrics

from file_loader import FileLoader
from data_filter import DataFilter
from feature_extractor import FeatureExtractor
from classifier import Classifier
import numpy as np
from feature_selector import Features
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score


SUBJECT = 'P5'
SUBJECTS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
NAME = '_READ'
NAMES = ['_READ', '_WRITE', '_WATCH', '_PLAY', '_BROWSE']
LOCATION = ".\\data\\DesktopActivity\\" + SUBJECT + "\\" + SUBJECT + NAME + ".csv"
DISTANCE_ARRAY = np.load('data\\' + SUBJECT + "\\" + NAME + '.npy')
FEATURES = [e for e in Features]
print('Features: ', FEATURES)


if __name__ == '__main__':
    data_X_train = []
    data_Y_train = []
    data_X_test = []
    data_Y_test = []

    for subject in SUBJECTS:
        print(subject)

        for name in NAMES:
            print(name)

            dist = np.load('data\\' + subject + "\\" + name + '.npy')
            location = ".\\data\\DesktopActivity\\" + subject + "\\" + subject + name + ".csv"

            fl = FileLoader()
            data = fl.read_file(location)
            data_raw = fl.normalize_data(np.asarray(data))

            df = DataFilter(data_raw, dist)
            filtered = np.asarray(df.median_filter_with_distance_array(df.raw_data, 6))
            fixations = np.asarray(df.get_fixations(filtered, 8, 0.003, 0.001))

            fe = FeatureExtractor(filtered, df.peak_indices, fixations)

            fts, labels = fe.windowed_features(1800, 1800*0.9, FEATURES, name)

            if not fts[0]:
                raise Exception("Features are empty, check features")

            train_X, test_X, train_y, test_y = train_test_split(fts, labels, test_size=0.2, random_state=0)
            data_X_train += train_X
            data_Y_train += train_y
            data_X_test += test_X
            data_Y_test += test_y

    svm = svm.SVC()
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(max_depth=4, random_state=0)
    et = ExtraTreesClassifier(n_estimators=100, random_state=0)

    clf = Classifier(svm, knn, rf)

    print(clf.svm.get_params())
    print(clf.knn.get_params())
    print(clf.rf.get_params())
    print(et.get_params())

    grid = {'C': [1, 10, 100, 1000], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']}
    scoring = ['accuracy', 'f1_macro']

    result = GridSearchCV(clf.svm, grid, scoring=scoring, refit='accuracy', cv=5)
    result.fit(data_X_train, data_Y_train)
    print(result.get_params(deep=False))
    print(accuracy_score(result.predict(data_X_test), data_Y_test))

    # print("KNN: ", clf.cross_validate(clf.knn, data_X_train + data_X_test, data_Y_train + data_Y_test))
    # print("SVM: ", clf.cross_validate(clf.svm, data_X_train + data_X_test, data_Y_train + data_Y_test))
    # print("Random Forest: ", clf.cross_validate(clf.rf, data_X_train + data_X_test, data_Y_train + data_Y_test))
    # print("Extra Trees: ", clf.cross_validate(et, data_X_train + data_X_test, data_Y_train + data_Y_test))
