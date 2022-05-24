from file_loader import FileLoader
from data_filter import DataFilter
from feature_extractor import FeatureExtractor
import numpy as np
import matplotlib.pyplot as plt

subject = "subject1"
name = 'writeDataRaw'
desktop_activity_location = ".\\data\\DesktopActivity\\" + subject + ".mat"

if __name__ == '__main__':
    fl = FileLoader()
    distance_array = np.load('data\\' + subject + "\\" + name + '.npy')
    data = fl.read_file(desktop_activity_location)
    data_raw = fl.normalize_data(np.asarray(data[name])[0:5000])
    df = DataFilter(data_raw, distance_array)

    filtered = np.asarray(df.median_filter_with_distance_array(df.raw_data, 90))
    fixations = np.asarray(df.get_fixations(df.filtered_data, 5, 0.001, 0.001))

    fe = FeatureExtractor(filtered, fixations)
    fixation_durations = fe.fixation_duration(df.peak_indices)
    saccade_lengths = fe.saccade_length(df.fixations)

    print(fe.saccade_length_avg_var_sd(df.fixations))
    #fe.plot_features(fixation_durations, "Fixation duration " + subject + " " + name, "Fixation", "Duration (ms)")
    fe.plot_features(saccade_lengths, "Saccade length " + subject + " " + name, "Saccade", "Length")
    #df.plot_raw_data_with_fixations(df.raw_data, df.fixations, name + ' ' + subject)
    #df.plot_fixations(df.fixations, name + ' ' + subject)
    #print(df.filtered_data[df.peak_indices[14]:df.peak_indices[15]])
    #fl.position_over_time(data_raw)
    #fl.position_over_time(filtered)



