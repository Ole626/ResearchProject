from file_loader import FileLoader
from data_filter import DataFilter
from feature_extractor import FeatureExtractor
from classifier import Classifier
import numpy as np


SUBJECT = 'subject1'
SUBJECTS = ['subject1', 'subject2', 'subject3', 'subject5', 'subject6', 'subject7', 'subject8']
NAME = 'readDataRaw'
NAMES = ['readDataRaw', 'writeDataRaw', 'playDataRaw', 'browseDataRaw', 'watchDataRaw']
LOCATION = ".\\data\\DesktopActivity\\" + SUBJECT + ".mat"
DISTANCE_ARRAY = np.load('data\\' + SUBJECT + "\\" + NAME + '.npy')


if __name__ == '__main__':
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []

    for subject in SUBJECTS[0:5]:
        for name in NAMES:
            dist = np.load('data\\' + subject + "\\" + name + '.npy')
            location = ".\\data\\DesktopActivity\\" + subject + ".mat"
            fl = FileLoader()
            data = fl.read_file(location)
            data_raw = fl.normalize_data(np.asarray(data[name])[0:10000])

            df = DataFilter(data_raw, dist)

            # filtered = np.asarray(df.median_filter_with_distance_array(df.raw_data, 6))
            fixations = np.asarray(df.get_fixations(data_raw, 8, 0.003, 0.001))

            # df.plot_fixations(fixations[0:6], 'Fixations')
            # fl.position_over_time(data_raw)
            # fl.position_over_time(filtered)

            fe = FeatureExtractor(None, df.peak_indices)

            saccade_directions = fe.saccade_direction(fixations)
            fixation_durations = fe.fixation_duration(df.peak_indices)
            saccade_lengths = fe.saccade_length(df.fixations)

            features = [fe.fixation_duration_avg_var_sd(df.peak_indices), fe.saccade_length_avg_var_sd(df.fixations)]

            train_X.append(list(sum(features, ())))
            train_Y.append(name)

    for subject in SUBJECTS[5:7]:
        for name in NAMES:
            dist = np.load('data\\' + subject + "\\" + name + '.npy')
            location = ".\\data\\DesktopActivity\\" + subject + ".mat"
            fl = FileLoader()
            data = fl.read_file(location)
            data_raw = fl.normalize_data(np.asarray(data[name])[0:10000])

            df = DataFilter(data_raw, dist)

            # filtered = np.asarray(df.median_filter_with_distance_array(df.raw_data, 6))
            fixations = np.asarray(df.get_fixations(data_raw, 8, 0.003, 0.001))

            # df.plot_fixations(fixations[0:6], 'Fixations')
            # fl.position_over_time(data_raw)
            # fl.position_over_time(filtered)

            fe = FeatureExtractor(None, df.peak_indices)

            saccade_directions = fe.saccade_direction(df.fixations)
            fixation_durations = fe.fixation_duration(df.peak_indices)
            saccade_lengths = fe.saccade_length(df.fixations)

            features = [fe.fixation_duration_avg_var_sd(df.peak_indices), fe.saccade_length_avg_var_sd(df.fixations)]

            test_X.append(list(sum(features, ())))
            test_Y.append(name)

    clf = Classifier()
    clf.svm_learn(train_X, train_Y)
    print(train_X)
    print(train_Y)
    print(test_X)
    print(test_Y)

    print(clf.svm_accuracy(test_X, test_Y))


    # fe.plot_features(saccade_directions, 'Saccade Directions ' + NAME + ' ' + SUBJECT, 'Saccade', 'Direction')
    # fe.plot_features(fixation_durations, 'Fixation Duration ' + NAME + ' ' + SUBJECT, 'Fixation', 'Duration (ms)')
    # fe.plot_features(saccade_lengths, 'Saccade Length ' + NAME + ' ' + SUBJECT, 'Saccade', 'Length')
    # print(fe.fixation_duration_avg_var_sd(df.peak_indices))
