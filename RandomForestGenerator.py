# -*- coding: utf-8 -*-
""" This module contains functionality for building a random forest classifier object and functions to build the forest
from a specified number of trees.
"""


class RandomForestClassifier(object):
    """
    :param  nb_trees:       Number of decision trees to use
    :param  nb_samples:     Number of samples to give to each tree
    :param  max_depth:      Maximum depth of the trees
    :param  max_workers:    Maximum number of processes to use for training
    """

    def __init__(self, nb_trees, nb_samples, max_depth=-1, max_workers=1):
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.max_workers = max_workers
