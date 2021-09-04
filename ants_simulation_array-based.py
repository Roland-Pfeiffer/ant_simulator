#!/usr/env/bin python3

import numpy as np

class Ants():
    def __init__(self, map_size, ant_count=500, coordinates=None, orientations=None):
        """

        :param ant_count:
        :type ant_count: int
        :param map_size: (width, height)
        :type map_size: tuple
        :param coordinates: 2D array, containing x and y coordinates
        :type coordinates:
        :param orientations:
        :type orientations:
        """
        if coordinates is not None:
            self.coordinates = coordinates
        else:
            self.coordinates = np.zeros([2, ant_count])


a = Ants(map_size=(400, 200))
print(a.coordinates)