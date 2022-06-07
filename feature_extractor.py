import matplotlib.pyplot as plt
import numpy as np
import math

POINT_DURATION = 100/3  # In ms


# The FeatureExtractor class takes preprocessed that and extracts features from the fixations and saccades.
# It is able to plot these as well.
class FeatureExtractor:
    def __init__(self, filtered_data, peak_indices):
        self.filtered_data = filtered_data
        self.peak_indices = peak_indices
        self.fixation_durations = None
        self.saccade_lengths = None
        self.saccade_directions = None

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

    def saccade_direction(self, fixations):
        deltaX = 0
        deltaY = 0

        result_directions = []

        for i in range(1, len(fixations)):
            deltaX = fixations[i][0] - fixations[i-1][0]
            deltaY = fixations[i][1] - fixations[i-1][1]
            length = np.sqrt(math.pow(deltaX, 2) + math.pow(deltaY, 2))
            normalized_deltaX = deltaX / length
            normalized_deltaY = deltaY / length

            if 0 <= normalized_deltaX < 0.5 and normalized_deltaY >= 0:
                result_directions.append(1)
            elif normalized_deltaX >= 0.5 and normalized_deltaY >= 0:
                result_directions.append(2)
            elif normalized_deltaX >= 0.5 and normalized_deltaY < 0:
                result_directions.append(3)
            elif 0 <= normalized_deltaX < 0.5 and normalized_deltaY < 0:
                result_directions.append(4)
            elif -0.5 <= normalized_deltaX < 0 and normalized_deltaY < 0:
                result_directions.append(5)
            elif normalized_deltaX < -0.5 and normalized_deltaY < 0:
                result_directions.append(6)
            elif normalized_deltaX < -0.5 and normalized_deltaY >= 0:
                result_directions.append(7)
            elif -0.5 <= normalized_deltaX < 0 and normalized_deltaY >= 0:
                result_directions.append(8)

        self.saccade_directions = result_directions
        return self.saccade_directions

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