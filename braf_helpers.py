# -*- coding: utf-8 -*-
""" This module contains helper functions to help execute the braf algorithm in main().
"""
import math
import numpy as np
import pandas as pd
from confusion_helpers import dict_list_appender


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
    """""
    Calculates the k nearest neighbors to a specific row in an array.
    :param dataset: numpy array.
    :param row: numpy array.
    :param k_neighbors: integer specifying the number of rows to return.
    :return: numpy array of nearest neighbors.
    """
    distances = list()
    for train_row in dataset:
        dist = euclidean_distance(row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(0, k_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def calc_unique_neighbors(full_dataset, k, T_min):
    """
    Calculate the unique k nearest neighbors of each row in T_min in full_dataset.
    :param full_dataset: pandas dataframe.
    :param k: int.
    :param T_min: pandas dataframe.
    :return: numpy list of unique k nearest neighbors.
    """
    T_min_nearest_neighbors = []

    for i in range(0, len(T_min)):
        T_min_nearest_neighbors.append(get_neighbors(full_dataset, T_min.iloc[i].values, k))
    T_min_nearest_neighbors_flat = [item for sublist in T_min_nearest_neighbors for item in sublist]
    return np.unique(T_min_nearest_neighbors_flat, axis=0)


def calculate_model_metrics(training_data, model):
    """
    This function records precision, recall, and training_outcome probabilities for the random forest braf algorithm.
    :param training_data: pandas dataframe.
    :param model: RandomForestClassifier object.
    :return: float, float, float, dict, dict representing
    precision, recall, false_positive_rate, metrics_dict_trees, metrics_dict the later two of which are data structures
    to hold precision, recall, training_outcomes and probabilities of those training_outcomes determined by the trees
    in the random forest.
    """

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    # Get list of features from the dataset.
    features = [ft[:-1] for ft in training_data.values]
    # Get list of values from the dataset (the last column).
    values = [ft[-1] for ft in training_data.values]

    # Initialize empty data structures to hold precision, recall, training outcomes and probability values.
    metrics_dict = {'precision': [], 'recall': []}
    metrics_dict_trees = {'training_outcomes': [], 'probabilities': []}

    for feature, value in zip(features, values):

        # Get the prediction from the model, as well as the precision and recall from the individual decision trees.
        prediction, tree_metrics, metrics = model.predict(feature, value)
        metrics_dict_trees = dict_list_appender(metrics_dict_trees, tree_metrics)
        metrics_dict = dict_list_appender(metrics_dict, metrics)

        # From the prediction made by the random forest, compare it to the actual value stored in values and determine
        # if the prediction was a false positive, true positive, true negative, or false negative.
        if prediction != value:
            if prediction == 1 and value == 0:
                false_positive += 1
            else:
                false_negative += 1
        elif prediction == 0:
            true_negative += 1
        else:
            true_positive += 1

    # Calculate precision, recall, and false positive rate.
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    false_positive_rate = false_positive / (false_positive + true_negative)

    return precision, recall, false_positive_rate, metrics_dict_trees, metrics_dict


def oversampler(oversample_factor, T_min, T_maj, training_data):
    # Reformulate training_data_minus_fold to oversample the minority class instances.
    num_minority_class_samples = int(oversample_factor * len(training_data))
    num_majority_class_samples = int((1 - oversample_factor) * len(training_data))
    training_data_maj_oversampled = T_maj.sample(num_majority_class_samples, replace=True)
    training_data_min_oversampled = T_min.sample(num_minority_class_samples, replace=True)
    frames = [training_data_maj_oversampled, training_data_min_oversampled]
    training_data_oversampled = pd.concat(frames)
    training_data_oversampled = training_data_oversampled.sample(frac=1)

    return training_data_oversampled
