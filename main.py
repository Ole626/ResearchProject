from file_loader import FileLoader
from data_filter import DataFilter
from feature_extractor import FeatureExtractor
from classifier import Classifier
import numpy as np
from feature_selector import Features
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, preprocessing
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt


SUBJECT = 'P5'
SUBJECTS = ['P1']
NAME = '_READ'
NAMES = ['_WATCH']
LOCATION = ".\\data\\DesktopActivity\\" + SUBJECT + "\\" + SUBJECT + NAME + ".csv"
DISTANCE_ARRAY = np.load('data\\' + SUBJECT + "\\" + NAME + '.npy')
FEATURES = [e for e in Features if e.value[0] not in []]
print('Features: ', FEATURES)


if __name__ == '__main__':
    data_X = []
    data_Y = []
    data_X_per_subject = []
    data_Y_per_subject = []

    for subject in SUBJECTS:
        print(subject)
        data_X_act = []
        data_Y_act = []

        for i, name in enumerate(NAMES):
            print(name)

            dist = np.load('data\\' + subject + "\\" + name + '.npy')
            location = ".\\data\\DesktopActivity\\" + subject + "\\" + subject + name + ".csv"

            fl = FileLoader()
            data = fl.read_file(location)
            data_raw = fl.normalize_data(np.asarray(data))

            df = DataFilter(data_raw, dist)
            filtered = np.asarray(df.median_filter_with_distance_array(df.raw_data, 6))
            df.plot_fixations(data_raw, "Raw Data Watching Subject 1")
            df.plot_fixations(filtered, "Filtered Data Watching Subject 1")
            fixations = np.asarray(df.get_fixations(filtered, 7, 0.003, 0.001))

            fe = FeatureExtractor(filtered, df.peak_indices, fixations)

            fts, labels = fe.windowed_features(1800, 1800*0.9, FEATURES, name)

            if not fts[0]:
                raise Exception("Features are empty, check features")
            data_X_act.append(fts)
            data_Y_act.append(labels)
            data_X += fts
            data_Y += labels

        data_X_per_subject.append(data_X_act)
        data_Y_per_subject.append(data_Y_act)

    scaler = preprocessing.StandardScaler().fit(data_X)
    scaled_subject_X = []
    for temp_sub in data_X_per_subject:
        scaled_act_X = []
        for temp_act in temp_sub:
            scaled_act_X.append(list(scaler.transform(temp_act)))
        scaled_subject_X.append(scaled_act_X)
    data_X = scaler.transform(data_X)

    svm = svm.SVC(C=1000, kernel='rbf')
    knn = KNeighborsClassifier(n_neighbors=20)
    rf = RandomForestClassifier()

    clf = Classifier(svm, knn, rf)

    clss = [(rf, 'Random Forest:'), (svm, 'SVM:')]

    for cls, name in clss:
        results, estimators = clf.cross_validate_over_activities(cls, scaled_subject_X, data_Y_per_subject)
        # results, estimators = clf.cross_validate_over_subjects(cls, data_X, data_Y)
        print(name, results)

        # feats = []
        # for est in estimators:
        #     feats.append(est.feature_importances_)

    features = ["Fixation Rate", "Fixation Duration Avg", "Fixation Duration Var", "Fixation Duration Std",
                "Saccade Length Avg", "Saccade Length Var", "Saccade Length Std", "Saccade Direction NNE",
                "Saccade Direction ENE", "Saccade Direction ESE", "Saccade Direction SSE", "Saccade Direction SSW",
                "Saccade Direction WSW", "Saccade Direction WNW", "Saccade Direction NNW", "Fixation Dispersion Area",
                "Fixation Slope", "Fixation Radius"]

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot()
    # zipped = sorted(list(zip(abs(np.average(np.asarray(feats), axis=0)), features)), reverse=True)
    # ax.bar([tup[1] for tup in zipped], [tup[0] for tup in zipped])
    # plt.xticks(rotation='vertical')
    # plt.tight_layout()
    # plt.show()
    # fig.savefig(".\\plots\\coefficients_svm.png")
