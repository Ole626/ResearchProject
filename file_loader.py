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

    def scatter_plot(self, xy_coordinates, name):
        transposed_array = xy_coordinates.T
        plt.plot(transposed_array[0], transposed_array[1])
        plt.title(name)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

