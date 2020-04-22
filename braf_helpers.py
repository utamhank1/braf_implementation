# -*- coding: utf-8 -*-
""" This module contains helper functions to help execute the braf algorithm in main().
"""
import math
import numpy as np


def euclidean_distance(row1, row2):
    """
    Calculates the Euclidean distance between two rows.
    :param row1: numpy array.
    :param row2: numpy array.
    :return: float representing the jaccard distance from row1 to row2.
    """
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


def get_neighbors(dataset, row, k_neighbors):
    '''
    Calculates the k nearest neighbors to a specific row in an array.
    :param dataset: numpy array.
    :param row: numpy array.
    :param k_neighbors: integer specifying the number of rows to return.
    :return: numpy array of nearest neighbors.
    '''
    distances = list()
    for train_row in dataset:
        dist = euclidean_distance(row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def calc_unique_neighbors(full_dataset, k, T_min):
    T_min_nearest_neighbors = []

    for i in range(0, len(T_min)):
        T_min_nearest_neighbors.append(get_neighbors(full_dataset, T_min.iloc[i].values, k))
    T_min_nearest_neighbors_flat = [item for sublist in T_min_nearest_neighbors for item in sublist]
    return np.unique(T_min_nearest_neighbors_flat, axis=0)
