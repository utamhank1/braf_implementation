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

    """
        Recursively creates the decision tree by splitting the dataset until no
        gain of information is added, or until the max depth is reached.
        :param  rows:   The dataset
        :param  func:   The function used to calculate the best split and stop
                        condition
        :param  depth:  The current depth in the tree
        """

    def build_tree(self, rows, func, depth):
        """
        Builds a decision tree recursively by splitting the dataset until there is no additional information gain or
        the maximum specified depth is reached.
        :param rows: list of lists representing the dataset.
        :param func: function used for calculating stop and split conditions.
        :param depth: The depth of the tree (current).
        :return: DecisionNode object.
        """

        # Base case.
        if len(rows) == 0:
            return self.DecisionNode()
        if depth == 0:
            return self.DecisionNode(results=unique_counts(rows))

        current_score = func(rows)
        best_gain = 0.0
        best_criteria = None
        best_sets = None
        column_count = len(rows[0]) - 1

        # Build Tree branches for every column.
        for col in range(0, column_count):
            column_values = {}
            for row in rows:
                column_values[row[col]] = 1
            for value in column_values.keys():
                set1, set2 = divide_set(rows, col, value)

                p = float(len(set1)) / len(rows)
                gain = current_score - p * func(set1) - (1 - p) * func(set2)
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)

        # Recursion.
        if best_gain > 0:
            trueBranch = self.build_tree(best_sets[0], func, depth - 1)
            falseBranch = self.build_tree(best_sets[1], func, depth - 1)
            return self.DecisionNode(col=best_criteria[0],
                                     value=best_criteria[1],
                                     tb=trueBranch, fb=falseBranch)
        else:
            return self.DecisionNode(results=unique_counts(rows))

    """
        :param  rows:       The data used to rain the decision tree. It must be a
                            list of lists. The last vaue of each inner list is the
                            value to predict.
        :param  criterion:  The function used to split data at each node of the
                            tree. If None, the criterion used is entropy.
        """

    def fit(self, rows, criterion=None):
        """
        This function trains the decision tree based on the supplied data.
        :param rows: list of lists representing the dataset supplied.
        :param criterion: Function supplied for the purposes of splitting data at each node in a tree. Default is the
                          entropy function.
        """
        if len(rows) < 1:
            raise ValueError("Not enough samples in the given dataset")

        if not criterion:
            # Use entropy function as a default.
            criterion = entropy
        if self.random_features:
            self.features_indexes = choose_random_features(rows[0])
            rows = [self.get_features_subset(row) + [row[-1]] for row in rows]

        self.root_node = self.build_tree(rows, criterion, self.max_depth)

    def classify(self, observation, tree):
        """
        Recursive function that makes a prediction on the label based on the supplied features.
        :param observation: list of features used for prediction.
        :param tree: dict object representing current node.on.
        """
        if tree.results is not None:
            return list(tree.results.keys())[0]
        else:
            v = observation[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.classify(observation, branch)

    def predict(self, features):
        """
        Returns a prediction for the features given.
        :param features: list.
        :return: integer representing the prediction.
        """
        if self.random_features:
            if not all(i in range(len(features))
                       for i in self.features_indexes):
                raise ValueError("Mismatch between the given features and those of the training set")
            features = self.get_features_subset(features)

        return self.classify(features, self.root_node)