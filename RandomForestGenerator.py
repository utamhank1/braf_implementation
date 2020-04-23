# -*- coding: utf-8 -*-
""" This module contains functionality for building a random forest classifier object and functions to build the forest
from a specified number of trees.
"""
import random
from concurrent.futures import ProcessPoolExecutor
from DecisionTree import DecisionTreeClassifier


class RandomForestClassifier(object):

    def __init__(self, nb_trees, nb_samples, max_depth=-1, max_workers=1):
        """
        :param  nb_trees:       Number of decision trees to use
        :param  nb_samples:     Number of samples to give to each tree
        :param  max_depth:      Maximum depth of the trees
        :param  max_workers:    Maximum number of processes to use for training
        """
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.max_workers = max_workers

    """
        Trains a single tree and returns it.
        :param  data:   A List containing the index of the tree being trained
                        and the data to train it
        """

    def train_tree(self, data):
        """
        Trains a singular tree and returns that tree.
        :param data: list representing the tree index being trained and the dataset being used for training.
        :return: DecisionNodeClassifier object.
        """
        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        tree.fit(data[1])
        return tree

    def fit(self, data):
        """
        Trains the number of decision trees based on self.nb_trees.
        :param data: list of lists representing the dataset with the last column in each inner list being the prediction
                     value.
        """
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            rand_fts = map(lambda x: [x, random.sample(data, self.nb_samples)],
                           range(self.nb_trees))
            self.trees = list(executor.map(self.train_tree, rand_fts))
