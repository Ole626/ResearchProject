import math
import matplotlib.pyplot as plt
import numpy as np


class DataFilter:
    def __init__(self, xy_coordinates):
        self.fixations = None
        self.saccades = None
        self.xy_coordinates = xy_coordinates
        self.threshold = 0.001

    def get_fixations(self):
        self.fixations = []
        current_fixation = []
        in_fixation = False

        for i in range(0, len(self.xy_coordinates) - 2):
            dist = self.euclidean_distance(self.xy_coordinates[i], self.xy_coordinates[i + 1])
            if not in_fixation and dist <= self.threshold:
                in_fixation = True
                current_fixation.append(self.xy_coordinates[i])
            if in_fixation:
                if dist <= self.threshold:
                    current_fixation.append(self.xy_coordinates[i])
                else:
                    current_fixation.append(self.xy_coordinates[i])
                    in_fixation = False
                    self.fixations.append(self.average_point(current_fixation))
                    current_fixation = []

        return self.fixations

    def euclidean_distance(self, coordinate1, coordinate2):
        return math.sqrt(math.pow((coordinate1[0] - coordinate2[0]), 2)
                         + math.pow((coordinate1[1] - coordinate2[1]), 2))

    def average_point(self, point_list):
        x_cumulative = 0
        y_cumulative = 0

        for point in point_list:
            x_cumulative = x_cumulative + point[0]
            y_cumulative = y_cumulative + point[1]

        return x_cumulative / len(point_list), y_cumulative / len(point_list)

    def plot_fixations(self):
        rawdata = np.asarray(self.xy_coordinates)
        fixations = np.asarray(self.fixations)
        plt.plot(rawdata.T[0], rawdata.T[1], c="blue")
        plt.scatter(fixations.T[0], fixations.T[1], c="red")
        plt.plot(fixations.T[0], fixations.T[1], c="red")
        plt.title("Test")
        plt.show()

