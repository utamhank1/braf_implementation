# -*- coding: utf-8 -*-
""" This module contains the steps for execution of the braf algorithm as detailed in the paper
"Biased Random Forest For Dealing With the Class Imbalance Problem".
"""

import pandas as pd
from braf_helpers import calc_unique_neighbors, calculate_model_metrics
from RandomForestGenerator import RandomForestClassifier


def braf(training_data, test_data, s, p, K):
    """
    This function executes the braf algorithm as detailed in the paper
    "Biased Random Forest For Dealing With the Class Imbalance Problem".
    :param training_data: pandas dataframe.
    :param test_data: pandas dataframe.
    :param s: int
    :param p: float
    :param K: int
    :return: float, float, float, dict, dict representing
    precision, recall, false_positive_rate, metrics_dict_trees, metrics_dict the later two of which are data structures
    to hold precision, recall, training_outcomes and probabilities of those training_outcomes determined by the trees
    in the random forest.
    """

    # Step a, extract T_min minority class from training dataset.
    T_min = training_data.loc[training_data['Outcome'] == 1].reset_index(drop=True)

    # Step b, isolate "difficult areas" affecting the minority instances.
    # For each record in T_min, create the find the k-nearest neighbors, save these nearest neighbors in T_c.
    training_data_minus_fold_values = training_data.values
    k_nearest_neighbors = 2
    T_c = pd.DataFrame(calc_unique_neighbors(training_data_minus_fold_values, k_nearest_neighbors, T_min),
                       columns=training_data.columns)
    # Step c, build the main random forest rf classifier from the full dataset.
    rf = RandomForestClassifier(nb_trees=int((1 - p) * s), nb_samples=K, max_workers=4)

    # Step d, Append the random forest generated from the dataset of the critical areas and specify size of the random
    # forest generated from the critical dataset.
    rf.fit_combined(list(training_data_minus_fold_values), list(T_c.values), nb_trees_2=int(s * p))

    # Calculate metrics from model.
    return calculate_model_metrics(test_data, model=rf)

