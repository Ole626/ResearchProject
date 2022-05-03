import math
import matplotlib.pyplot as plt
import numpy as np


class DataFilter:
    def __init__(self, xy_coordinates):
        self.fixations = None
        self.saccades = None
        self.xy_coordinates = xy_coordinates
        self.threshold = 0.001
        self.outlier_dist = 0.4

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
                    self.fixations.append(self.mean_point(current_fixation))
                    current_fixation = []

        return self.fixations

    def euclidean_distance(self, coordinate1, coordinate2):
        return math.sqrt(math.pow((coordinate1[0] - coordinate2[0]), 2)
                         + math.pow((coordinate1[1] - coordinate2[1]), 2))

    def mean_point(self, point_list):
        x_cumulative = 0
        y_cumulative = 0

        for point in point_list:
            x_cumulative = x_cumulative + point[0]
            y_cumulative = y_cumulative + point[1]

        return x_cumulative / len(point_list), y_cumulative / len(point_list)

    def remove_outliers(self, xy_coordinates):
        for i in range(0, math.floor(len(xy_coordinates)/500)):
            mean_point = self.mean_point(xy_coordinates[i*500:(i+1)*500])
            for point in xy_coordinates[i*500:(i+1)*500]:
                dist = self.euclidean_distance(point, mean_point)
                if dist > self.outlier_dist:
                    xy_coordinates.remove(point)
        return xy_coordinates


    def plot_fixations(self):
        rawdata = np.asarray(self.xy_coordinates)
        fixations = np.asarray(self.fixations)
        plt.plot(rawdata.T[0], rawdata.T[1], c="blue")
        plt.scatter(fixations.T[0], fixations.T[1], c="red")
        plt.plot(fixations.T[0], fixations.T[1], c="red")
        plt.title("Test")
        plt.show()

