import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter
from feature_selector import Features
import sys


POINT_DURATION = 100/3  # In ms


# The FeatureExtractor class takes preprocessed that and extracts features from the fixations and saccades.
# It is able to plot these as well.
class FeatureExtractor:
    def __init__(self, filtered_data, peak_indices, fixations):
        self.filtered_data = filtered_data
        self.peak_indices = peak_indices
        self.fixations = fixations
        self.fixation_durations = None
        self.saccade_lengths = None
        self.saccade_directions = None

    # This function calculates the fixation rate
    def fixation_rate(self, fixations):
        return len(fixations) / (len(self.filtered_data) * POINT_DURATION / 1000)

    # This function calculates the duration of every fixation.
    def fixation_duration(self, peak_indices):
        durations = []
        for i in range(1, len(peak_indices)):
            durations.append((peak_indices[i] - peak_indices[i-1]) * POINT_DURATION)

        self.fixation_durations = durations
        return self.fixation_durations

    # This function calculates the average, variation and standard deviation of the fixation duration.
    def fixation_duration_avg_var_sd(self, peak_indices):
        durations = self.fixation_duration(peak_indices)
        average = np.average(durations)
        variance = np.var(durations)
        std = np.std(durations)

        return average, variance, std

    # This function calculates the saccade length.
    def saccade_length(self, fixations):
        lengths = []
        for i in range(1, len(fixations)):
            lengths.append(self.euclidean_distance(fixations[i-1], fixations[i]))

        self.saccade_lengths = lengths
        return self.saccade_lengths

    # This function calculates the average, variation and standard deviation of the saccade length.
    def saccade_length_avg_var_sd(self, fixations):
        lengths = self.saccade_length(fixations)
        average = np.average(lengths)
        variance = np.var(lengths)
        std = np.std(lengths)

        return average, variance, std

    # This function returns the sum of all the directions the saccades are directed split into 8 directions.
    def saccade_direction(self, fixations):
        deltaX = 0
        deltaY = 0

        result_directions = {'NNE': 0, 'ENE': 0, 'ESE': 0, 'SSE': 0, 'SSW': 0, 'WSW': 0, 'WNW': 0, 'NNW': 0}

        for i in range(1, len(fixations)):
            deltaX = fixations[i][0] - fixations[i-1][0]
            deltaY = fixations[i][1] - fixations[i-1][1]
            length = np.sqrt(math.pow(deltaX, 2) + math.pow(deltaY, 2))
            normalized_deltaX = deltaX / length
            normalized_deltaY = deltaY / length

            if 0 <= normalized_deltaX < 0.5 and normalized_deltaY >= 0:
                result_directions['NNE'] += 1
            elif normalized_deltaX >= 0.5 and normalized_deltaY >= 0:
                result_directions['ENE'] += 1
            elif normalized_deltaX >= 0.5 and normalized_deltaY < 0:
                result_directions['ESE'] += 1
            elif 0 <= normalized_deltaX < 0.5 and normalized_deltaY < 0:
                result_directions['SSE'] += 1
            elif -0.5 <= normalized_deltaX < 0 and normalized_deltaY < 0:
                result_directions['SSW'] += 1
            elif normalized_deltaX < -0.5 and normalized_deltaY < 0:
                result_directions['WSW'] += 1
            elif normalized_deltaX < -0.5 and normalized_deltaY >= 0:
                result_directions['WNW'] += 1
            elif -0.5 <= normalized_deltaX < 0 and normalized_deltaY >= 0:
                result_directions['NNW'] += 1

        self.saccade_directions = [quotient / (len(fixations)-1) for quotient in Counter(result_directions).values()]
        return self.saccade_directions

    # This function returns the dispersion area of a list of fixations after removing some outliers from the data.
    def fixation_dispersion_area(self, fixations):
        distances = []
        mean_point = np.average(fixations, axis=0)

        for fixation in fixations:
            distances.append(self.euclidean_distance(fixation, mean_point))

        zipped = list(zip(distances, fixations))
        zipped.sort(key=lambda tup: tup[0], reverse=True)
        sorted_points = [list(point[1]) for point in zipped[0:int(0.75 * len(fixations))]]
        min_max_x, min_max_y = self.min_max(sorted_points)

        return (min_max_x[1] - min_max_x[0]) * (min_max_y[1] - min_max_y[0])

    # This function returns the radius to the furthest fixation
    def fixation_radius(self, fixations):
        distances = []
        mean_point = np.average(fixations, axis=0)

        for fixation in fixations:
            distances.append(self.euclidean_distance(fixation, mean_point))

        zipped = list(zip(distances, fixations))
        zipped.sort(key=lambda tup: tup[0], reverse=True)

        return zipped[0][0]

    # This function returns the slope of the best fit line through the fixations
    def fixation_slope(self, fixations):
        transposed = np.asarray(fixations).T
        slope_and_b = np.polyfit(transposed[0], transposed[1], 1)
        return slope_and_b[0]

    # This function calculates all selected features of 1 activity with a window and overlap.
    def windowed_features(self, window_size, overlap, features, label):
        data_x = []
        data_y = []

        for i in range(0, int(len(self.filtered_data) - window_size), int(window_size-overlap)):
            feature_list = []
            sliced_peaks = [j for j in range(0, len(self.peak_indices)) if i <= self.peak_indices[j] <= i+window_size]
            sliced_fixations = self.fixations[sliced_peaks[0]:sliced_peaks[-1]]

            for feature in features:
                match feature:
                    case Features.FIX_RATE:
                        feature_list += [self.fixation_rate(sliced_fixations)]
                    case Features.FIX_DUR:
                        feature_list += [ft for ft in self.fixation_duration_avg_var_sd(sliced_peaks)]
                    case Features.SAC_LEN:
                        feature_list += [ft for ft in self.saccade_length_avg_var_sd(sliced_fixations)]
                    case Features.SAC_DIR:
                        feature_list += self.saccade_direction(sliced_fixations)
                    case Features.FIX_DIS:
                        feature_list += [self.fixation_dispersion_area(sliced_fixations)]
                    case Features.FIX_SLOPE:
                        feature_list += [self.fixation_slope(sliced_fixations)]
                    case Features.FIX_RADIUS:
                        feature_list += [self.fixation_radius(sliced_fixations)]

            data_x.append(feature_list)
            data_y.append(label)

        return data_x, data_y

    # This function plots a given list of features in a bar plot.
    def plot_features(self, features, name, x_label, y_label):
        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_subplot()

        ax.bar(np.arange(len(features)), features)
        ax.set_title(name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(0, np.max(features) * 1.2)

        plt.tight_layout()
        plt.show()
        fig.savefig(".\\plots\\" + name + ".png")

    # This function calculates the euclidean distance of two points.
    def euclidean_distance(self, coordinate1, coordinate2):
        return math.sqrt(math.pow((coordinate1[0] - coordinate2[0]), 2)
                         + math.pow((coordinate1[1] - coordinate2[1]), 2))

    # This function returns the max and min of both the x and y coordinates.
    def min_max(self, xy_coordinates):
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
