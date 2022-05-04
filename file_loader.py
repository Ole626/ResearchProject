import scipy.io as sp
import sys
import numpy as np
import matplotlib.pyplot as plt


class FileLoader:
    def __init__(self):
        self.x_coordinates = None
        self.y_coordinates = None

    def read_file(self, file_location):
        data = sp.loadmat(file_location)
        return data

    def get_min_max(self, xy_coordinates):
        min_x = sys.maxsize
        max_x = -sys.maxsize
        min_y = sys.maxsize
        max_y = -sys.maxsize

        for coordinate in xy_coordinates:
            if coordinate[0] < min_x:
                min_x = coordinate[0]
            if coordinate[0] > max_x:
                max_x = coordinate[0]
            if coordinate[1] < min_y:
                min_y = coordinate[1]
            if coordinate[1] > max_y:
                max_y = coordinate[1]

        return (min_x, max_x), (min_y, max_y)

    def normalize_data(self, xy_coordinates):
        minmax = self.get_min_max(xy_coordinates)
        minmax_x = minmax[0]
        minmax_y = minmax[1]
        normalized_coordinates = []

        for coordinate in xy_coordinates:
            normalized_coordinate = ((coordinate[0] - minmax_x[0])/(minmax_x[1] - minmax_x[0]),
                                     (coordinate[1] - minmax_y[0])/(minmax_y[1] - minmax_y[0]))
            normalized_coordinates.append(normalized_coordinate)

        return normalized_coordinates

    def scatter_plot(self, xy_coordinates, name):
        transposed_array = np.asarray(xy_coordinates).T
        plt.plot(transposed_array[0], transposed_array[1], linewidth=0.3, color='black', marker='o',
                 markerfacecolor='red', markeredgecolor='black', markeredgewidth=1.2, markersize=4.0)
        plt.title(name)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def position_over_time(self, xy_coordinates):
        transposed_array = np.asarray(xy_coordinates).T
        time = np.arange(len(xy_coordinates)) * 20
        plt.plot(time, transposed_array[0], color='red', label='x')
        plt.plot(time, transposed_array[1], color='blue', label='y')
        plt.legend(loc='upper right')
        plt.xlabel('Time (ms)')
        plt.ylabel('Position')
        plt.show()
