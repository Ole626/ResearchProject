import matplotlib.pyplot as plt
import numpy as np
import math

POINT_DURATION = 100/3  # In ms


class FeatureExtractor:
    def __init__(self, filtered_data, peak_indices):
        self.filtered_data = filtered_data
        self.peak_indices = peak_indices
        self.fixation_durations = None
        self.saccade_lengths = None

    def fixation_duration(self, peak_indices):
        durations = []
        for i in range(1, len(peak_indices)):
            durations.append((peak_indices[i] - peak_indices[i-1]) * POINT_DURATION)

        self.fixation_durations = durations
        return self.fixation_durations

    def fixation_duration_avg_var_sd(self, peak_indices):
        durations = self.fixation_duration(peak_indices)
        average = np.average(durations)
        variance = np.var(durations)
        std = np.std(durations)

        return average, variance, std

    def saccade_length(self, fixations):
        lengths = []
        for i in range(1, len(fixations)):
            lengths.append(self.euclidean_distance(fixations[i-1], fixations[i]))

        self.saccade_lengths = lengths
        return self.saccade_lengths

    def saccade_length_avg_var_sd(self, fixations):
        lengths = self.saccade_length(fixations)
        average = np.average(lengths)
        variance = np.var(lengths)
        std = np.std(lengths)

        return average, variance, std


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