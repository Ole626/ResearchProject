from file_loader import FileLoader
from data_filter import DataFilter
import numpy as np
import matplotlib.pyplot as plt


desktop_activity_location = ".\\data\\DesktopActivity\\subject7.mat"

if __name__ == '__main__':
    fl = FileLoader()
    data = fl.read_file(desktop_activity_location)
    name = 'searchDataRaw'
    data_raw = np.asarray(data[name][0:500])
    df = DataFilter(data_raw)
    median_point = df.geometric_median(data_raw)
    filtered = np.asarray(df.median_filter(data_raw, 11))
    fixations = np.asarray(df.get_fixations(filtered, 5, 0.01, 1))
    plt.plot(filtered.T[0], filtered.T[1], color='blue')
    plt.scatter(fixations.T[0], fixations.T[1], color='red')
    plt.show()
    #fl.position_over_time(data_raw)
    #fl.position_over_time(filtered)



