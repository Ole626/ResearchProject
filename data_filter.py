import math
import sys
import matplotlib.pyplot as plt
import numpy as np


# The DataFilter class can preprocess raw data by applying a median filter, extract fixations and plot the results.
class DataFilter:
    def __init__(self, raw_data, distance_array):
        self.fixations = None
        self.saccades = None
        self.raw_data = raw_data
        self.filtered_data = None
        self.distance_array = distance_array
        self.threshold = 0.001
        self.peak_indices = None

    # This function calculates fixations using the fixation filter from Pontus Olsson on the filtered data using a given
    # sliding window, threshold and radius.
    def get_fixations(self, xy_coordinates, sliding_window, threshold, radius):
        difference_vector = np.zeros(len(xy_coordinates))

        # Get the difference vector for the means of consecutive sliding windows.
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

        # Add only the peaks that go above the threshold.
        for i in range(0, len(peaks)-1):
            if peaks[i] >= threshold:
                peak_indices.append(i)

        # Determine positions of fixations by geometric median. Remove fixations that are too close to each other.
        shortest_distance = 0
        while shortest_distance < radius:
            self.fixations = []
            for i in range(1, len(peak_indices)-1):
                fixation = self.geometric_median(xy_coordinates[peak_indices[i-1]:peak_indices[i]])
                self.fixations.append(fixation)

            shortest_distance = sys.maxsize
            index = 0
            for i in range(1, len(self.fixations)):
                distance = self.euclidean_distance(self.fixations[i], self.fixations[i-1])
                if distance < shortest_distance:
                    shortest_distance = distance
                    index = i
            if shortest_distance < radius:
                del peak_indices[index]
        self.peak_indices = peak_indices
        return self.fixations

    # This function calculates the euclidean distance of two points.
    def euclidean_distance(self, coordinate1, coordinate2):
        return math.sqrt(math.pow((coordinate1[0] - coordinate2[0]), 2)
                         + math.pow((coordinate1[1] - coordinate2[1]), 2))

    # This function applies a median filter over self.raw_data with a given sliding window calculating the distances
    # for each sliding window every time.
    def median_filter(self, xy_coordinates, sliding_window):
        output_coordinates = []
        counter = 0
        # Loop through all sliding windows and add the median to output_coordinates.
        for i in range(sliding_window, len(xy_coordinates) - sliding_window):
            window = []
            for j in range(0, sliding_window):
                window.append(i+j-sliding_window)

            # if (i-sliding_window) % int(((len(xy_coordinates) - (2 * sliding_window)) / 100)) == 0:
            #     counter = counter + 1
            #     print(counter, "%")

            output_coordinates.append(self.geometric_median(window))

        self.filtered_data = output_coordinates
        return self.filtered_data

    # This function applies the median filter over self.raw_data with a given sliding window using precalculated
    # distances to improve performance.
    def median_filter_with_distance_array(self, xy_coordinates, sliding_window):
        output_coordinates = []
        counter = 0
        for i in range(sliding_window, len(xy_coordinates) - sliding_window):
            window = []
            # if (i-sliding_window) % int(((len(xy_coordinates) - 2 * sliding_window) / 100)) == 0:
            #     counter = counter + 1
            #     print(counter, "%")
            for j in range(0, sliding_window):
                window.append(i+j-sliding_window)

            output_coordinates.append(
                self.raw_data[self.geometric_median_with_distance_array(window)])

        self.filtered_data = output_coordinates
        return self.filtered_data

    # This function calculates the geometric median using the euclidean distance of a list of coordinates.
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

    # This function calculates the geometric median using a precomputed matrix of distances and a given list of indices
    # corresponding to this matrix.
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

    # This function precomputes all distances from self.raw_data and saves it in data/*subject*/*name*.
    def calculate_all_distances(self, name, subject):
        result_array = np.ndarray(shape=(len(self.raw_data), len(self.raw_data)), dtype=float)
        counter = 0
        for i in range(0, len(self.raw_data)):
            for j in range(i+1, len(self.raw_data)):
                result_array[i][j] = self.euclidean_distance(self.raw_data[i], self.raw_data[j])

            # if (i-1) % (len(self.raw_data) / 100) == 0:
            #     counter = counter + 1
            #     print(counter, "%")
        np.save('data\\' + subject + "\\" + name + '.npy', result_array)
        return result_array

    # This function plots the fixations in self.fixations in a plot.
    def plot_fixations(self, fixations, name):
        fig = plt.figure(figsize=(5, 4))
        fixations = np.asarray(fixations)

        ax = fig.add_subplot()

        ax.plot(fixations.T[0], fixations.T[1], linewidth=0.8, color='black', marker='o',
                 markerfacecolor='red', markeredgecolor='black', markeredgewidth=1.5, markersize=4.5)
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

    # This function plots the raw data with the fixations on top.
    def plot_raw_data_with_fixations(self, raw_data, fixations, name):
        fig = plt.figure(figsize=(5, 4))
        rawdata = np.asarray(raw_data)
        fixations = np.asarray(fixations)

        ax = fig.add_subplot()

        ax.plot(rawdata.T[0], rawdata.T[1], color='blue', alpha=0.5, linewidth=2)
        ax.scatter(fixations.T[0], fixations.T[1], color='red', edgecolor='black', linewidths=1.5)
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()
