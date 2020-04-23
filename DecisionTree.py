# -*- coding: utf-8 -*-
""" This module contains functionality for building a decision tree with a DecisionNodeClassifier object along with
several functions to train the classifier.
"""


class DecisionTreeClassifier(object):
    class DecisionNode(object):
        def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
            self.col = col
            self.value = value
            self.results = results
            self.tb = tb
            self.fb = fb

        """
       :param  max_depth:          Represents max number of training splits
       :param  random_features:    If not true, use all features to train and predict. If True, use sqrt(nb features)
       """

    def __init__(self, max_depth=-1, random_features=False):
        self.root_node = None
        self.max_depth = max_depth
        self.features_indexes = []
        self.random_features = random_features
