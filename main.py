from file_loader import FileLoader
from data_filter import DataFilter
import numpy as np


desktop_activity_location = ".\\data\\DesktopActivity\\subject5.mat"

if __name__ == '__main__':
    fl = FileLoader()
    data = fl.read_file(desktop_activity_location)
    normalized = fl.normalize_data(data['readDataRaw'])
    name = 'writeDataRaw'
    data_raw = data[name]
    df = DataFilter(data_raw)
    filtered = df.median_filter(data_raw, 400)
    print(len(filtered))
    print(len(data_raw))
    fl.scatter_plot(data_raw, "Raw: " + name)
    fl.scatter_plot(filtered, "Filtered: " + name)




