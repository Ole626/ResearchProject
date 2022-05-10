from file_loader import FileLoader
from data_filter import DataFilter
import numpy as np
import matplotlib.pyplot as plt


desktop_activity_location = ".\\data\\DesktopActivity\\subject5.mat"

if __name__ == '__main__':
    name = 'readDataRaw'
    fl = FileLoader()
    data = fl.read_file(desktop_activity_location)
    distance_array = np.load('data\\' + name + '.npy')
    data_raw = np.asarray(data[name])
    #plt.plot(data_raw.T[0], data_raw.T[1])
    #plt.show()
    normalized = fl.normalize_data(data_raw)

    df = DataFilter(normalized[0:5000], distance_array)
    filtered = np.asarray(df.median_filter(90))

    fixations = np.asarray(df.get_fixations(filtered, 4, 0.005, 0.001))
    plt.plot(fixations.T[0], fixations.T[1], linewidth=0.6, color='black', marker='o',
             markerfacecolor='red', markeredgecolor='black', markeredgewidth=1.5, markersize=4.5)
    plt.show()
    #fl.position_over_time(data_raw)
    #fl.position_over_time(filtered)



