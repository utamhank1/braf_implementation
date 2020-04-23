# -*- coding: utf-8 -*-
""" This module contains functionality for building a decision tree with a DecisionNodeClassifier object along with
several functions to train the classifier.
"""
import random
from math import log, sqrt


def choose_random_features(row):
    """
    This function chooses random features in a supplied list.
    :param row: list.
    :return: list of random features selected.
    """
    nb_features = len(row) - 1
    return random.sample(range(nb_features), int(sqrt(nb_features)))


def unique_counts(rows):
    """
    Counts the number of each result in the dataset.
    :param rows: list of lists with label at the end of each inner list.
    :return: dict of counts of occurrence of each result.
    """
    results = {}
    for row in rows:
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


def entropy(rows):
    """
    Returns the entropy value in the rows of the dataset given.
    :param rows: list of lists representing the datset supplied (must have label as the last column os each
                inner list).
    :return: float representing the entropy value.
    """
    results = unique_counts(rows)
    log2 = lambda x: log(x) / log(2)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


def divide_set(rows, column, value):
    """
    Splits the dataset based on the value at a specified column index.
    :param rows: list of lists representing the dataset.
    :param column: int column index used to base where to split the data.
    :param value: float The value that is used in the split.
    :return: set, set based on the two pieces of the split dataset.
    """
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]

    return set1, set2


class DecisionTreeClassifier(object):
    class DecisionNode(object):
        def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
            self.col = col
            self.value = value
            self.results = results
            self.tb = tb
            self.fb = fb

    def __init__(self, max_depth=-1, random_features=False):
        """
        :param  max_depth: int representing max number of training splits.
        :param  random_features: Boolean If True, use sqrt(nb features).
                                If not true, use all features to train and predict.
        """
        self.root_node = None
        self.max_depth = max_depth
        self.features_indexes = []
        self.random_features = random_features

    def get_features_subset(self, row):
        """
        Returns values that are randomly selected in the given features.
        :param row: list.
        :return: list of randomly selected values.
        """
        return [row[i] for i in self.features_indexes]
