from file_loader import FileLoader
import numpy as np


desktop_activity_location = ".\\data\\DesktopActivity\\subject5.mat"

if __name__ == '__main__':
    fl = FileLoader()
    data = fl.read_file(desktop_activity_location)

    for key, value in data.items():
        if 'DataRaw' in key:
            fl.scatter_plot(value, key)
            print(key, fl.get_min_max(data[key]))
