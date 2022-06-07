import scipy.io as sp
import sys
import numpy as np
import matplotlib.pyplot as plt


# The FileLoader class reads the data and normalizes the data. It can also plot the data.
class FileLoader:
    def __init__(self):
        self.x_coordinates = None
        self.y_coordinates = None

    # This function loads .mat files into an array.
    def read_file(self, file_location):
        data = np.genfromtxt(file_location, delimiter=',')
        return data

    # This function returns the max and min of both the x and y coordinates.
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

    # This function normalizes the data given into a range between 0 and 1.
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

    # This function plots all the individual points in a scatter plot.
    def scatter_plot(self, xy_coordinates, name):
        transposed_array = np.asarray(xy_coordinates).T
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot()

        ax.plot(transposed_array[0], transposed_array[1], linewidth=0.3, color='black', marker='o',
                markerfacecolor='red', markeredgecolor='black', markeredgewidth=1.2, markersize=4.0)
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        plt.tight_layout()
        plt.show()

    # This function show the flow of the x and y coordinates of data over time.
    def position_over_time(self, xy_coordinates):
        transposed_array = np.asarray(xy_coordinates).T
        time = np.arange(len(xy_coordinates)) / 300

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot()

        ax.plot(time, transposed_array[0], color='red', label='x')
        ax.plot(time, transposed_array[1], color='blue', label='y')
        ax.legend(loc='upper right')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position')

        plt.tight_layout()
        plt.show()
