import math

class DataFilter:
    def __init__(self, xy_coordinates):
        self.fixations = None
        self.saccades = None
        self.xy_coordinates = xy_coordinates
        self.threshold = None

    def get_fixations(self):


    def euclidean_distance(self, coordinate1, coordinate2):
        return math.sqrt(math.pow((coordinate1[0] - coordinate2[0]),2) + math.pow((coordinate1[1] - coordinate2[1]),2))
