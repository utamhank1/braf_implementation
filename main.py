import argparse
import os
import sys
import pandas as pd
import scipy
from scipy import stats as sps
import numpy as np
import data_explorer
import data_preprocessor
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from braf_helpers import get_neighbors, calc_unique_neighbors, calculate_model_metrics
import math
from RandomForestGenerator import RandomForestClassifier
import braf_main
from confusion_statistics_helpers import dict_list_appender
import pdb


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
    training_data_master = data[int(.2 * len(data)):len(data)]

    # Separate labels from the rest of the dataset
    holdout_data_labels = holdout_data['Outcome'].copy()
    holdout_data = holdout_data.drop('Outcome', axis=1)

    # TODO: Add K as an input to argparser.
    K = 10

    shuffled_data = training_data_master.sample(frac=1)

    # Create K random divisions of the test data and store them in a pandas dataframe.
    K_folds = pd.DataFrame(np.array_split(shuffled_data, K))

    number_folds = len(K_folds)

    ####################################################################################################################
    ############################################ BRAF Algorithm. #######################################################
    ####################################################################################################################

    # TODO: Add these parameters as inputs to argparser.
    p = .5
    s = 100
    metrics_dict = {'precision': [], 'recall': [], 'FPR': []}
    metrics_dict_tree_master = {'precision': [], 'recall': [], 'FPR': []}

    for i in range(0, 1):

        # Remove the first 1/10 of the data in the k-folds cross validation from the training dataset.
        training_data_minus_fold = training_data_master.drop(K_folds[0][i].index)

        # # Calculate metrics from model.
        run_metrics = braf_main.braf(training_data=training_data_minus_fold, test_data=K_folds[0][i], s=s, p=p, K=K)

        metrics_dict = dict_list_appender(metrics_dict, run_metrics[:-1])

        for key in metrics_dict_tree_master.keys():
            metrics_dict_tree_master[key] = metrics_dict_tree_master[key] + run_metrics[-1][key]

        # print(f"Precision = {precision}")
        # print(f"Recall = {recall}")
        # print(f"FPR = {false_positive_rate}")
    # metrics_dict = pd.read_csv('metrics_for_10_folds.csv')
    print(metrics_dict_tree_master)
    # plt.plot(metrics_dict['FPR'], metrics_dict['recall'], c='g', linewidth=4)
    # plt.xlabel('False Positive Rate', fontsize=16)
    # plt.ylabel('True Positive Rate', fontsize=16)
    # plt.title('Receiver Operating Characteristic', fontsize=16)
    # plt.legend(loc='lower right', fontsize=16)

    # metrics_dataframe = pd.DataFrame(metrics_dict)

    #metrics_dataframe.to_csv(f'metrics_for_{K}_folds_iter3.csv')

if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
