from file_loader import FileLoader
from data_filter import DataFilter
import numpy as np


desktop_activity_location = ".\\data\\DesktopActivity\\subject5.mat"

if __name__ == '__main__':
    fl = FileLoader()
    data = fl.read_file(desktop_activity_location)
    fl.scatter_plot(data['readDataRaw'], "Read Raw")
    normalized = fl.normalize_data(data['readDataRaw'])
    fl.scatter_plot(normalized, "Read Normalized")



