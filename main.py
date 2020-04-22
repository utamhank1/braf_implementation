import argparse
import os
import sys
import pandas as pd
import scipy
from scipy import stats
import numpy as np
import data_explorer
import data_preprocessor
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from braf_helpers import get_neighbors, calc_unique_neighbors
from scipy.spatial import KDTree
import math


def parse_arguments():
    """
    Function that parses user set command line arguments to pass to program.
    :return: string
    """
    # Infrastructure to set up command line argument parser.
    parser = argparse.ArgumentParser(description='Enter the path to the directory of the PIMA diabetes.csv file you '
                                                 'wish to analyze'
                                     )
    parser.add_argument("-f", "--file", type=str, help='Path to the directory containing the .txt files'
                                                       'DIRECTORY MUST NOT CONTAIN ANY SPACES IN FOLDER NAMES')

    args = parser.parse_args()
    file = args.file

    # Input validation for file.
    if not os.path.exists(file):
        print('The file specified does not exist or is not in the current directory.')
        sys.exit()

    return file


def main(file):
    print(f"Hello, World the file supplied is {file}")

    ####################################################################################################################
    ########################################### Data Importation. ######################################################
    ####################################################################################################################

    raw_data = pd.DataFrame(pd.read_csv(file))
    # print(raw_data.head())

    ####################################################################################################################
    ############################################ Data Exploration. #####################################################
    ####################################################################################################################

    # # Draw correlation heatmap for all features.
    # plt.figure(figsize=(10, 10))
    # plt.show(data_preprocessor.data_explorer(raw_data).draw_correlations())
    #
    # # Draw histograms of distributions of features for people with and without diabetes.
    # data_preprocessor.data_explorer(raw_data).draw_distributions()
    # plt.show()
    #
    # # Save summary statistics for each feature.
    # feature_summary_statistics = data_preprocessor.data_explorer(raw_data).print_summary()

    ####################################################################################################################
    ############################################ Data Pre-processing. ##################################################
    ####################################################################################################################

    # TODO: Generate all of the processed_data objects.
    # std_dev_to_keep = [2.5, 2.75, 3.0]
    # imputation_methods = ['random', 'mean', 'median']
    std_dev_to_keep = [2.75]
    imputation_methods = ['random']
    processed_data_objects = collections.defaultdict(list)

    # # Create nine preprocessed and imputed data objects with varying std. deviations kept and imputation methods.
    for imputation_method in imputation_methods:
        for std_dev in std_dev_to_keep:
            processed_data_objects[("data_std_dev_" + str(std_dev).replace('.', '_') + "_impute_" +
                                    str(imputation_method))].append(data_preprocessor.
                                                                    preprocessed_data(raw_data, stdev_to_keep=std_dev).
                                                                    impute(imputation_method=imputation_method))

    ####################################################################################################################
    ############################################ Data Splitting. #######################################################
    ####################################################################################################################

    # TODO: Iterate through all of the objects in processed_data objects.
    data = processed_data_objects['data_std_dev_2_75_impute_random'][0]

    # 80/20 test split for training and holdout data.
    holdout_data = data[0:int(.2 * len(data))]
    training_data = data[int(.2 * len(data)):len(data)]

    # TODO: Add K as an input to argparser.
    K = 10

    shuffled_data = training_data.sample(frac=1)

    # Create K random divisions of the test data and store them in a pandas dataframe.
    K_folds = pd.DataFrame(np.array_split(shuffled_data, K))

    for i in range(0, len(K_folds)):
        K_folds[0][i] = K_folds[0][i].reset_index(drop=True)
    ####################################################################################################################
    ############################################ BRAF Algorithm. #######################################################
    ####################################################################################################################

    # TODO: Add these parameters as inputs to argparser.
    p = .05
    s = 100

    # Step a, split into T_maj and T_min majority/minority classes.
    T_maj = K_folds[0][0].loc[K_folds[0][0]['Outcome'] == 0].reset_index(drop=True)
    T_min = K_folds[0][0].loc[K_folds[0][0]['Outcome'] == 1].reset_index(drop=True)

    # Step b, isolate "difficult areas" affecting the minority instances.
    # For each record in T_min, create the find the k-nearest neighbors, save these nearest neighbors in T_c.
    full_dataset = K_folds[0][0].values
    k = int(math.sqrt(len(full_dataset)))
    T_c = pd.DataFrame(calc_unique_neighbors(full_dataset, k, T_min), columns=raw_data.columns)

    # Step c, build the random forest RF1 from the full dataset.
    RF1 = build_random_forest


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
