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

    def median_filter(self, xy_coordinates, sliding_window):
        output_coordinates = []
        for i in range(sliding_window, len(xy_coordinates) - sliding_window):
            window = []
            for j in range(0, sliding_window):
                window.append(xy_coordinates[i+j-sliding_window])

            avg_point = self.mean_point(window)
            distances = []
            for entry in window:
                distances.append(self.euclidean_distance(entry, avg_point))

            point_dist = list(zip(window, distances))
            point_dist.sort(key=lambda tup: tup[1])
            output = point_dist[int(np.floor(sliding_window/2))][0]

            in_array = False
            for coordinate in output_coordinates:
                if coordinate[0] == output[0] and coordinate[1] == output[1]:
                    in_array = True
                    break
            if not in_array:
                output_coordinates.append(output)

        return output_coordinates


    def plot_fixations(self):
        rawdata = np.asarray(self.xy_coordinates)
        fixations = np.asarray(self.fixations)
        plt.plot(rawdata.T[0], rawdata.T[1], c="blue")
        plt.scatter(fixations.T[0], fixations.T[1], c="red")
        plt.plot(fixations.T[0], fixations.T[1], c="red")
        plt.title("Test")
        plt.show()

