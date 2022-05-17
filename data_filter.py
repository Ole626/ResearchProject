import math
import sys

import matplotlib.pyplot as plt
import numpy as np


class DataFilter:
    def __init__(self, xy_coordinates, distance_array):
        self.fixations = None
        self.saccades = None
        self.xy_coordinates = xy_coordinates
        self.distance_array = distance_array
        self.threshold = 0.001
        self.outlier_dist = 0.4
        self.fixation_lengths = None

    def get_fixations(self, xy_coordinates, sliding_window, threshold, radius):
        difference_vector = np.zeros(len(xy_coordinates))

        # Get the difference vector for the mean of consecutive sliding windows.
        for i in range(sliding_window, len(xy_coordinates)-sliding_window):
            m_before = [0, 0]
            m_after = [0, 0]

            for j in range(1, sliding_window):
                m_before[0] = m_before[0] + xy_coordinates[i-j][0]/sliding_window
                m_before[1] = m_before[1] + xy_coordinates[i-j][1]/sliding_window
                m_after[0] = m_after[0] + xy_coordinates[i+j][0]/sliding_window
                m_after[1] = m_after[1] + xy_coordinates[i+j][1]/sliding_window

            difference_vector[i] = np.sqrt(math.pow((m_after[0] - m_before[0]), 2) +
                                           math.pow((m_after[1] - m_before[1]), 2))

        peaks = np.zeros(len(xy_coordinates))
        print("Difference vector done")
        # Get the peaks between averages and store them in a separate list.
        for i in range(1, len(xy_coordinates)-1):
            if difference_vector[i] > difference_vector[i-1] and difference_vector[i] > difference_vector[i+1]:
                peaks[i] = difference_vector[i]

        # Remove peaks that lie within the same sliding windows.
        for i in range(sliding_window, len(xy_coordinates)-sliding_window):
            if peaks[i] != 0:
                for j in range(i-sliding_window, i-1):
                    if peaks[j] <= peaks[i]:
                        peaks[j] = 0
                for j in range(i+1, i+sliding_window):
                    if peaks[j] <= peaks[i]:
                        peaks[j] = 0

        peak_indices = []

        for i in range(0, len(peaks)-1):
            if peaks[i] >= threshold:
                peak_indices.append(i)
        print("Determined peak indices")

        # Determine positions of fixations by geometric median. Remove fixations that are too close to each other.
        shortest_distance = 0
        while shortest_distance < radius:
            self.fixations = []
            self.fixation_lengths = np.zeros(len(peak_indices)-2)
            for i in range(1, len(peak_indices)-1):
                fixation = self.geometric_median(xy_coordinates[peak_indices[i-1]:peak_indices[i]])
                self.fixations.append(fixation)
                for point in xy_coordinates[peak_indices[i-1]:peak_indices[i]]:
                    if self.euclidean_distance(point, fixation) <= 0.005:
                        self.fixation_lengths[i-1] = self.fixation_lengths[i-1] + 1/(peak_indices[i]-peak_indices[i-1])
            shortest_distance = sys.maxsize
            index = 0
            for i in range(1, len(self.fixations)):
                distance = self.euclidean_distance(self.fixations[i], self.fixations[i-1])
                if distance < shortest_distance:
                    shortest_distance = distance
                    index = i
            if shortest_distance < radius:
                del peak_indices[index]

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

    def median_filter(self, sliding_window):
        output_coordinates = []
        counter = 0
        for i in range(sliding_window, len(self.xy_coordinates) - sliding_window):
            window = []
            if (i-sliding_window) % int(((len(self.xy_coordinates)-2*sliding_window)/100)) == 0:
                counter = counter + 1
                print(counter, "%")
            for j in range(0, sliding_window):
                window.append(i+j-sliding_window)

            output_coordinates.append(self.geometric_median(window))

        print("Median filter done")
        return output_coordinates

    def median_filter_with_distance_array(self, sliding_window):
        output_coordinates = []
        counter = 0
        for i in range(sliding_window, len(self.xy_coordinates) - sliding_window):
            window = []
            if (i-sliding_window) % int(((len(self.xy_coordinates)-2*sliding_window)/100)) == 0:
                counter = counter + 1
                print(counter, "%")
            for j in range(0, sliding_window):
                window.append(i+j-sliding_window)

            output_coordinates.append(
                self.xy_coordinates[self.geometric_median_with_distance_array(window)])

        print("Median filter done")
        return output_coordinates

    def geometric_median(self, xy_coordinates):
        shortest_distance = sys.maxsize
        median = []

        for point in xy_coordinates:
            distance_sum = 0
            for comparison_point in xy_coordinates:
                if distance_sum > shortest_distance:
                    break
                distance_sum = distance_sum + self.euclidean_distance(point, comparison_point)

            if distance_sum < shortest_distance:
                shortest_distance = distance_sum
                median = point

        return median

    def geometric_median_with_distance_array(self, indices):
        shortest_distance = sys.maxsize
        median = []

        for point in indices:
            distance_sum = 0
            for comparison_point in indices:
                if distance_sum > shortest_distance:
                    break
                if point < comparison_point:
                    distance_sum = distance_sum + self.distance_array[point][comparison_point]
                else:
                    distance_sum = distance_sum + self.distance_array[comparison_point][point]

            if distance_sum < shortest_distance:
                shortest_distance = distance_sum
                median = point

        return median

    def calculate_all_distances(self, name, subject):
        result_array = np.ndarray(shape=(len(self.xy_coordinates), len(self.xy_coordinates)), dtype=float)
        counter = 0
        for i in range(0, len(self.xy_coordinates)):
            if (i-1) % (len(self.xy_coordinates)/100) == 0:
                counter = counter + 1
                print(counter, "%")
            for j in range(i+1, len(self.xy_coordinates)):
                result_array[i][j] = self.euclidean_distance(self.xy_coordinates[i], self.xy_coordinates[j])
        np.save('data\\' + subject + "\\" + name + '.npy', result_array)
        return result_array

    def plot_fixations(self, name):
        fixations = np.asarray(self.fixations)
        plt.plot(fixations.T[0], fixations.T[1], linewidth=0.8, color='black', marker='o',
                 markerfacecolor='red', markeredgecolor='black', markeredgewidth=1.5, markersize=4.5)
        plt.title(name)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def plot_raw_data_with_fixations(self, name):
        rawdata = np.asarray(self.xy_coordinates)
        fixations = np.asarray(self.fixations)
        plt.plot(rawdata.T[0], rawdata.T[1], color='blue', alpha=0.5)
        plt.scatter(fixations.T[0], fixations.T[1], color='red', edgecolor='black', linewidths=1.5)
        plt.title(name)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


