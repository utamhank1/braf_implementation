# -*- coding: utf-8 -*-
""" This module contains functionality for building a decision tree with a DecisionNodeClassifier object along with
several functions to train the classifier.
"""


class DecisionTreeClassifier:
    class DecisionNode:
        def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
            self.col = col
            self.value = value
            self.results = results
            self.tb = tb
            self.fb = fb
