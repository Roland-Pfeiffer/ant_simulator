#!/usr/bin/env python3

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time

# ToDo: Have them be able to look into ALL directions, not just Up Down Left Right!


class Ant():
    def __init__(self, x=None, y=None, map_size=None, orientation=None, p_keep_orientation=0.97):
        """

        :param x:
        :type x:
        :param y:
        :type y:
        :param map_size:
        :type map_size:
        :param orientation:
        :type orientation:
        :param p_keep_orientation:
        :type p_keep_orientation:
        """
        # If no coords provided, select at random
        if x is None and y is None:
            self.x = random.randint(0, map_size[0] - 1)  # -1 because it's inclusive and e.g. 0-99 = 100
            self.y = random.randint(0, map_size[1] - 1)  # -1 because it's inclusive and e.g. 0-99 = 100
        else:
            self.x = x
            self.y = y
        # If no orientation provided, select at random
        if orientation is None:
            self.orientation = random.sample(['up', 'down', 'left', 'right'], 1)[0]
        else:
            self.orientation = orientation
        self.map_size = map_size
        self.p_keep_orientation = p_keep_orientation

    def get_possible_target_fields(self):
        logging.info(f'Current coordinate (x, y): {self.x, self.y}')
        logging.info(f'Current orientation: {self.orientation}')
        ant_moves = {'up': ((-1, -1), (0, -1), (1, -1)),
                     'down': ((-1, 1), (0, 1), (1, 1)),
                     'left': ((-1, -1), (-1, 0), (-1, 1)),
                     'right': ((1, -1), (1, 0), (1, 1))}  # Note that y INCREASES downward
        possible_moves = ant_moves[self.orientation]
        target_fields = []

        # Fill theoretical next fields based on orientation
        for x_change, ychange in possible_moves:
            new_target = (self.x + x_change, self.y + ychange)
            # Verify new target is within map bounds:
            if 0 <= new_target[0] <= (self.map_size[0] - 1) and 0 <= new_target[1] <= (self.map_size[1] - 1):
                # -1 was because map_size is not zero indexing, but the rest is.
                target_fields.append(new_target)
        # If no valid target field in the orientation was found:
        if len(target_fields) == 0:
            target_fields.append((self.x, self.y))
        logging.debug(f'Possible targets: {target_fields} - {type(target_fields)}')
        return target_fields

    def move(self):
        targets = self.get_possible_target_fields()
        logging.debug(f'Target fields: {targets} - {type(targets)}')
        self.x, self.y = random.sample(targets, 1)[0]  # [0], else the results ist still list
        logging.info(f'New position: {self.x, self.y}')

        # Check if ant changes orientation: # ToDo: update this when scent trails are implemented.
        if random.random() > self.p_keep_orientation:
            options = ['up', 'down', 'left', 'right']
            options.remove(self.orientation)
            self.orientation = random.sample(options, 1)[0]
            # This needed [0], else the results ist still list

        logging.info(f'New orientation: {self.orientation}')

    def __repr__(self):
        return f'Ant(x={self.x}, y={self.y}, map_size={self.map_size}, orientation={self.orientation}'


class Map():
    def __init__(self, ants, map_size, decay_rate=0.005):
        self.ants = ants
        self.map = np.zeros((map_size[1], map_size[0]))
        for ant in ants:
            logging.debug(f'New pos.: {ant.x, ant.y}')
            _x, _y = ant.x, ant.y
            self.map[_y, _x] = 1
        logging.info(f'Map shape: {self.map.shape}')
        self.decay_rate = decay_rate

    def show(self):
        plt.imshow(self.map, cmap='gray')
        plt.show()

    def save(self, fname):
        plt.imsave(fname, self.map, cmap='gray')

    def show_array(self):
        print(self.map)

    def next_gen(self):
        self.map = np.subtract(self.map, self.decay_rate, out=self.map, where=self.map > 0)
        for ant in self.ants:
            ant.move()
            self.map[ant.y, ant.x] = 1


def create_ants(ant_count, map_size):
    ant_list = []
    for i in range(ant_count):
        ant_list.append(Ant(map_size=map_size))
    logging.debug(f'Ants: {ant_list}')
    [logging.debug(ant.x, ant.y) for ant in ant_list]
    return ant_list


def run_simulation(ant_count, map_size, generations=None, plot_images=False, save_images=False, return_stack=False):
    print('Starting simulation.')
    ants = create_ants(ant_count, map_size)
    board = Map(ants, map_size)
    save_path = os.getcwd() + '/renders'

    if return_stack:
        print('WARNING: Saving board states in memory. This can slow down the code. ',
              'Set return_stack to False to increase speed.')
        generation_stack = board.map

    if plot_images:
        board.show()
    if save_images:
        try:
            os.mkdir(save_path)
        except FileExistsError:
            input(f'Directory {save_path} already exists.\nContaining files will be overwritten. Continue?')

            board.save('renders/gen_0.png')
    if return_stack:
        print('WARNING: Saving board states in memory. This can slow down the code. ',
              'Set return_stack to False to increase speed.')
        generation_stack = board.map

    for gen in range(1, generations):
        if gen % 10 == 0:
            print(f'Generation {gen} done.')

        board.next_gen()

        if return_stack:
            generation_stack = np.dstack((generation_stack, board.map))
        if save_images:
            # ToDo: Investigate speed improvement through threading!
            board.save(fname=f'renders/gen_{gen + 1}.png')
        if plot_images:
            board.show()
    if return_stack:
        return generation_stack


if __name__ == '__main__':

    import concurrent.futures
    import logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s]\t%(message)s')
    logging.disable()

    # stack = run_simulation(ant_count=150, map_size=(1000, 1000), generations=250, save_images=True)
    run_simulation(ant_count=250, map_size=(640, 480), generations=500, save_images=True)
