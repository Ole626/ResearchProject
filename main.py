from file_loader import FileLoader
from data_filter import DataFilter
from feature_extractor import FeatureExtractor
from classifier import Classifier
import numpy as np


SUBJECT = 'P5'
SUBJECTS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
NAME = '_PLAY'
NAMES = ['_READ', '_WRITE', '_PLAY', '_BROWSE', '_WATCH', '_SEARCH']
LOCATION = ".\\data\\DesktopActivity\\" + SUBJECT + "\\" + SUBJECT + NAME + ".csv"
# DISTANCE_ARRAY = np.load('data\\' + SUBJECT + "\\" + NAME + '.npy')

if __name__ == '__main__':
    for subject in SUBJECTS:
        print(subject)
        for name in NAMES:
            print(name)
            location = ".\\data\\DesktopActivity\\" + subject + "\\" + subject + name + ".csv"
            fl = FileLoader()
            data = fl.read_file(location)
            data_raw = fl.normalize_data(np.asarray(data))

            df = DataFilter(data_raw, None)
            df.calculate_all_distances(name, subject)


# if __name__ == '__main__':
#     train_X = []
#     train_Y = []
#     test_X = []
#     test_Y = []
#
#     for subject in SUBJECTS:
#         print(subject)
#         for name in NAMES:
#             print(name)
#             dist = np.load('data\\' + subject + "\\" + name + '.npy')
#             location = ".\\data\\DesktopActivity\\" + subject + ".mat"
#             fl = FileLoader()
#             data = fl.read_file(location)
#             data_raw = fl.normalize_data(np.asarray(data[name])[0:20000])
#
#             df = DataFilter(data_raw, dist[0:20000, 0:20000])
#
#             filtered = np.asarray(df.median_filter_with_distance_array(df.raw_data, 6))
#             fixations = np.asarray(df.get_fixations(filtered, 8, 0.003, 0.001))
#
#             # df.plot_fixations(fixations[0:6], 'Fixations')
#             # fl.position_over_time(data_raw)
#             # fl.position_over_time(filtered)
#
#             fe = FeatureExtractor(filtered, df.peak_indices)
#
#             saccade_directions = fe.saccade_direction(fixations)
#             fixation_durations = fe.fixation_duration(df.peak_indices)
#             saccade_lengths = fe.saccade_length(df.fixations)
#             fixation_rate = fe.fixation_rate(df.fixations)
#
#             features = [fe.fixation_duration_avg_var_sd(df.peak_indices), fe.saccade_length_avg_var_sd(df.fixations)]
#             flatten_features = list(sum(features, ()))
#             flatten_features.append(fixation_rate)
#
#             train_X.append(flatten_features)
#             train_Y.append(name)
#
#     for subject in SUBJECTS:
#         print(subject)
#         for name in NAMES:
#             print(name)
#             dist = np.load('data\\' + subject + "\\" + name + '.npy')
#             location = ".\\data\\DesktopActivity\\" + subject + ".mat"
#             fl = FileLoader()
#             data = fl.read_file(location)
#             data_raw = fl.normalize_data(np.asarray(data[name])[20000:27000])
#
#             df = DataFilter(data_raw, dist[20000:27000, 20000:27000])
#
#             filtered = np.asarray(df.median_filter_with_distance_array(df.raw_data, 6))
#             fixations = np.asarray(df.get_fixations(filtered, 8, 0.003, 0.001))
#
#             # df.plot_fixations(fixations[0:6], 'Fixations')
#             # fl.position_over_time(data_raw)
#             # fl.position_over_time(filtered)
#
#             fe = FeatureExtractor(filtered, df.peak_indices)
#
#             saccade_directions = fe.saccade_direction(df.fixations)
#             fixation_durations = fe.fixation_duration(df.peak_indices)
#             saccade_lengths = fe.saccade_length(df.fixations)
#             fixation_rate = fe.fixation_rate(df.fixations)
#
#             features = [fe.fixation_duration_avg_var_sd(df.peak_indices), fe.saccade_length_avg_var_sd(df.fixations)]
#             flatten_features = list(sum(features, ()))
#             flatten_features.append(fixation_rate)
#
#             test_X.append(flatten_features)
#             test_Y.append(name)
#
#     clf = Classifier()
#     clf.svm_learn(train_X, train_Y)
#
#     print(clf.svm_accuracy(test_X, test_Y))
#
#
#     # fe.plot_features(saccade_directions, 'Saccade Directions ' + NAME + ' ' + SUBJECT, 'Saccade', 'Direction')
#     # fe.plot_features(fixation_durations, 'Fixation Duration ' + NAME + ' ' + SUBJECT, 'Fixation', 'Duration (ms)')
#     # fe.plot_features(saccade_lengths, 'Saccade Length ' + NAME + ' ' + SUBJECT, 'Saccade', 'Length')
#     # print(fe.fixation_duration_avg_var_sd(df.peak_indices))
