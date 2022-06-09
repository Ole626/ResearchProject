from file_loader import FileLoader
from data_filter import DataFilter
from feature_extractor import FeatureExtractor
from classifier import Classifier
import numpy as np
from feature_selector import Features
from sklearn.model_selection import train_test_split


SUBJECT = 'P5'
SUBJECTS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
NAME = '_READ'
NAMES = ['_READ', '_WATCH', '_PLAY', '_WRITE', '_BROWSE']
LOCATION = ".\\data\\DesktopActivity\\" + SUBJECT + "\\" + SUBJECT + NAME + ".csv"
DISTANCE_ARRAY = np.load('data\\' + SUBJECT + "\\" + NAME + '.npy')
FEATURES = [e for e in Features if e.value[0] in [2, 3]]
print('Features: ', FEATURES)


if __name__ == '__main__':
    data_X = []
    data_Y = []

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

            data_X += fts
            data_Y += labels

    train_X, test_X, train_y, test_y = train_test_split(data_X, data_Y, test_size=0.4, random_state=0)
    clf = Classifier()
    clf.svm_learn(train_X, train_y)
    clf.knn_learn(train_X, train_y)

    print(test_y)
    print(list(clf.svm_predict(test_X)))
    print(clf.svm_accuracy(test_X, test_y))
    print(clf.knn_accuracy(test_X, test_y))
