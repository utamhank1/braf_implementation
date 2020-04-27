# -*- coding: utf-8 -*-
""" This module contains the argument parser that allows one to run the main() function from a windows or linux command
line.
"""
import argparse
import os
import sys


def parse_arguments():
    """
    Function that parses user set command line arguments to pass to program.
    :return: string, int, int, float
    """
    # Infrastructure to set up command line argument parser.
    parser = argparse.ArgumentParser(description="Enter the path to the directory of the PIMA diabetes.csv file you "
                                                 "wish to analyze."
                                     )
    parser.add_argument("-f", "--file", type=str, help='Path to the directory containing the .txt files'
                                                       'DIRECTORY MUST NOT CONTAIN ANY SPACES IN FOLDER NAMES.')

    parser.add_argument("-K", "--folds", type=int, help="Number of divisions to make in training data for K-Fold Cross "
                                                        "validation.")

    parser.add_argument("-s", "--num_trees", type=int, help="Number of decision trees to create in each random forest.")

    parser.add_argument("-p", "--proportion", type=float,
                        help="Proportion of data to sample from the critical dataset.")

    parser.add_argument("-imp", "--imputation_method", default='mean', type=str, help="Type of imputation method to "
                                                                                      "use on the data, default is "
                                                                                      "'random' where a random number "
                                                                                      "is assigned to missing values "
                                                                                      "in the dataset that are "
                                                                                      "sampled from the gaussian "
                                                                                      "distribution generated from "
                                                                                      "the mean and std. deviation of "
                                                                                      "that column. Other valid "
                                                                                      "methods are 'mean' where the "
                                                                                      "mean value of that column is "
                                                                                      "imputed to all of the missing "
                                                                                      "values, and 'median', "
                                                                                      "where the median value of that "
                                                                                      "column is imputed onto the "
                                                                                      "missing values.")

    parser.add_argument("-stdev", "--std_dev_to_keep", default=3.5, type=float, help="Value for the number of std. "
                                                                                     "deviations of each feature to "
                                                                                     "keep. Default is 3.5")

    parser.add_argument("-exp", "--explore_data", default='False', type=bool, help="Indicate whether the user desires "
                                                                                   "to view correlational matrix and "
                                                                                   "histograms for data feature "
                                                                                   "distribution for both positive "
                                                                                   "and negative outcomes, "
                                                                                   "Default is False")

    args = parser.parse_args()
    file = args.file
    K = args.folds
    s = args.num_trees
    p = args.proportion
    imputation_method = args.imputation_method
    stdev = args.std_dev_to_keep
    exp = args.explore_data

    # Input validation for file.
    if not os.path.exists(file):
        print('The file specified does not exist or is not in the directory specified.')
        sys.exit()

    return file, K, s, p, imputation_method, stdev, exp

