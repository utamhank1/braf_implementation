# -*- coding: utf-8 -*-
""" This module contains functionality for building a random forest classifier object and functions to build the forest
from a specified number of trees.
"""
import random
from concurrent.futures import ProcessPoolExecutor
from DecisionTree import DecisionTreeClassifier
import pdb
from confusion_statistics_helpers import confusion_calculator


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

    def predict(self, feature, value):
        """
        Returns a prediction value from the given features based on the value that gets the most "votes" (one from
        each decision tree).
        :param feature: list of prediction features.
        :param value: integer or float representing the true value.
        :return: value representing the prediction.
        """
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(feature))

        run_metrics_trees = list(confusion_calculator(predictions, value))
        return max(set(predictions), key=predictions.count), run_metrics_trees

    def fit_combined(self, data1, data2, nb_trees_2):
        """
        This function generates, combines and trains the random forest generated with a dataset data2
        with size nb_trees_2, with the current random forest generated from data1.
        :param data1: list The first dataset that you wish to generate a random forest of and train.
        :param data2: list The second dataset that you wish to combine and train the random forest of.
        :param nb_trees_2: the intended size of the second random forest associated with data2.
        """
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            list_data = [data1, data2]
            list_nb_trees = [self.nb_trees, nb_trees_2]
            for data, nb_trees in zip(list_data, list_nb_trees):
                rand_fts_data = map(lambda x: [x, random.sample(data, self.nb_samples)],
                                    range(nb_trees))
                # combined the trained random forests of each dataset together.
                self.trees += list(executor.map(self.train_tree, rand_fts_data))
