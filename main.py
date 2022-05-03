from file_loader import FileLoader
from data_filter import DataFilter
import numpy as np


desktop_activity_location = ".\\data\\DesktopActivity\\subject5.mat"

if __name__ == '__main__':
    fl = FileLoader()
    data = fl.read_file(desktop_activity_location)
    normalized = fl.normalize_data(data['readDataRaw'])
    df = DataFilter(normalized)
    filtered = df.remove_outliers(normalized)
    fl.scatter_plot(filtered, "Filtered")



