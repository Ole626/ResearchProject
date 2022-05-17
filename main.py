from file_loader import FileLoader
from data_filter import DataFilter
import numpy as np
import matplotlib.pyplot as plt

subject = "subject1"
name = 'readDataRaw'
desktop_activity_location = ".\\data\\DesktopActivity\\" + subject + ".mat"

if __name__ == '__main__':
    fl = FileLoader()
    distance_array = np.load('data\\' + subject + "\\" + name + '.npy')
    data = fl.read_file(desktop_activity_location)
    data_raw = fl.normalize_data(np.asarray(data[name])[0:5000])
    df = DataFilter(data_raw, distance_array)

    filtered = np.asarray(df.median_filter_with_distance_array(90))

    fixations = np.asarray(df.get_fixations(filtered, 7, 0.005, 0.01))
    df.plot_raw_data_with_fixations(name + ' ' + subject)
    print(df.fixation_lengths)
    #fixations = np.asarray(df.get_fixations(data_raw, 7, 0.005, 0.01))
    #df.plot_raw_data_with_fixations(name + ' ' + subject)
    #df.plot_fixations(name + ' ' + subject)
    #fl.position_over_time(data_raw)
    #fl.position_over_time(filtered)



