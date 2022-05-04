from file_loader import FileLoader
from data_filter import DataFilter
import numpy as np


desktop_activity_location = ".\\data\\DesktopActivity\\subject7.mat"

if __name__ == '__main__':
    fl = FileLoader()
    data = fl.read_file(desktop_activity_location)
    name = 'searchDataRaw'
    data_raw = data[name][0:10000]
    df = DataFilter(data_raw)
    filtered = df.median_filter(data_raw, 200)
    fl.position_over_time(data_raw)
    fl.position_over_time(filtered)



