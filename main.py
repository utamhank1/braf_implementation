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
from confusion_statistics_helpers import roc_curve, prc_curve, prc_roc_curve


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
    std_dev_to_keep = [3.5]
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
    data = processed_data_objects[
        f"data_std_dev_{str(std_dev_to_keep[0]).replace('.', '_')}_impute_{str(imputation_methods[0])}"][0]

    # 80/20 test split for training and holdout data.
    holdout_data = data[0:int(.2 * len(data))]
    training_data_master = data[int(.2 * len(data)):len(data)]

    # Separate labels from the rest of the dataset
    holdout_data_labels = holdout_data['Outcome'].copy()
    # holdout_data = holdout_data.drop('Outcome', axis=1)

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
    p = .7
    s = 100
    metrics_dict = {'precision': [], 'recall': [], 'FPR': []}
    metrics_dict_tree_master = {'training_outcomes': [], 'probabilities': []}

    # for i in range(0, len(K_folds[0])):
    for i in range(0, 1):

        # Remove the first 1/10 of the data in the k-folds cross validation from the training dataset.
        training_data_minus_fold = training_data_master.drop(K_folds[0][i].index)

        # # Calculate metrics from model.
        run_metrics = braf_main.braf(training_data=training_data_minus_fold, test_data=K_folds[0][i], s=s, p=p, K=K)

        metrics_dict = dict_list_appender(metrics_dict, run_metrics[:-1])
        for key in metrics_dict_tree_master.keys():
            metrics_dict_tree_master[key] = metrics_dict_tree_master[key] + run_metrics[-2][key]

    fpr, tpr, precision, auc_roc, auc_prc = prc_roc_curve(
        np.array(metrics_dict_tree_master['training_outcomes']), np.array(metrics_dict_tree_master['probabilities']))

    plt.plot(fpr, tpr, 'b')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f"Training data Performance ROC: AUC = {auc_roc}")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)

    plt.show()

    plt.plot(tpr, precision, 'b')
    plt.plot([0, .8], [1, .8], 'r--')
    plt.title(f"Training data Performance PRC: AUC = {auc_prc}")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)

    plt.show()

    ###################################################################################################################
    ############################################# Testing Data Metrics. ###############################################
    ###################################################################################################################

    metrics_dict_test = {'precision': [], 'recall': [], 'FPR': []}
    metrics_dict_tree_master_test = {'training_outcomes': [], 'probabilities': []}

    # Apply best model from training data (one with highest recall)
    best_index = metrics_dict['recall'].index(max(metrics_dict['recall']))
    print(f"best_index = {best_index}")

    # Train model on best k-fold of training data, and apply it to make predictions on test data.
    # # Calculate metrics from model.
    training_data_minus_fold_for_test = training_data_master.drop(K_folds[0][best_index].index)
    run_metrics_test = braf_main.braf(training_data=training_data_minus_fold_for_test, test_data=holdout_data,
                                      s=s, p=p, K=K)

    metrics_dict_test = dict_list_appender(metrics_dict_test, run_metrics_test[:-1])
    for key in metrics_dict_tree_master_test.keys():
        metrics_dict_tree_master_test[key] = metrics_dict_tree_master_test[key] + run_metrics_test[-2][key]

    fpr, tpr, precision, auc_roc, auc_prc = prc_roc_curve(np.array(metrics_dict_tree_master_test['training_outcomes']),
                                                          np.array(metrics_dict_tree_master_test['probabilities']))

    plt.plot(fpr, tpr, 'b')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f"Testing data Performance ROC: AUC = {auc_roc}")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)

    plt.show()

    plt.plot(tpr, precision, 'b')
    plt.plot([0, .8], [1, .8], 'r--')
    plt.title(f"Testing data Performance PRC: AUC = {auc_prc}")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)

    plt.show()

    print(pd.DataFrame(metrics_dict_test))


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
