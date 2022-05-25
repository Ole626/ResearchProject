from file_loader import FileLoader
from data_filter import DataFilter
from feature_extractor import FeatureExtractor
import numpy as np


SUBJECT = "subject3"
NAME = 'readDataRaw'
LOCATION = ".\\data\\DesktopActivity\\" + SUBJECT + ".mat"
DISTANCE_ARRAY = np.load('data\\' + SUBJECT + "\\" + NAME + '.npy')


if __name__ == '__main__':
    fl = FileLoader()
    data = fl.read_file(LOCATION)
    data_raw = fl.normalize_data(np.asarray(data[NAME])[0:5000])

    df = DataFilter(data_raw, DISTANCE_ARRAY)
    filtered = np.asarray(df.median_filter_with_distance_array(df.raw_data, 90))
    fixations = np.asarray(df.get_fixations(df.filtered_data, 5, 0.001, 0.001))

    fe = FeatureExtractor(filtered, df.peak_indices)
    fixation_durations = fe.fixation_duration(df.peak_indices)
    saccade_lengths = fe.saccade_length(df.fixations)

    fe.plot_features(fixation_durations, 'Fixation Duration ' + NAME + ' ' + SUBJECT, 'Fixation', 'Duration (ms)')
    fe.plot_features(saccade_lengths, 'Saccade Length ' + NAME + ' ' + SUBJECT, 'Saccade', 'Length')





