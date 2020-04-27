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
from confusion_statistics_helpers import dict_list_appender, prc_roc_curve, curve_generator
import pdb


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


def main(file, K, s, p, imputation_method, stdev, exp):
    print(f"Hello world, the file supplied is {file}")

    ####################################################################################################################
    ########################################### Data Importation. ######################################################
    ####################################################################################################################

    raw_data = pd.DataFrame(pd.read_csv(file))

    ####################################################################################################################
    ############################################ Data Exploration. #####################################################
    ####################################################################################################################

    if exp:
        # Draw correlation heatmap for all features.
        plt.figure(figsize=(10, 10))
        plt.show(data_preprocessor.data_explorer(raw_data).draw_correlations())

        # Draw histograms of distributions of features for people with and without diabetes.
        data_preprocessor.data_explorer(raw_data).draw_distributions()
        plt.show()

    else:
        pass

    ####################################################################################################################
    ############################################ Data Pre-processing. ##################################################
    ####################################################################################################################

    std_dev_to_keep = [stdev]
    imputation_methods = [imputation_method]

    # Generate preprocessed data object according to the std. deviation we wish to keep and the imputation method.
    processed_data_objects = data_preprocessor.gen_preprocessed_objects(imputation_methods, std_dev_to_keep, raw_data)

    ####################################################################################################################
    ############################################ Data Splitting. #######################################################
    ####################################################################################################################

    data = processed_data_objects[
        f"data_std_dev_{str(std_dev_to_keep[0]).replace('.', '_')}_impute_{str(imputation_methods[0])}"][0]

    # Shuffle the full dataset.
    data = data.sample(frac=1)

    # 80/20 test split for training and holdout data.
    holdout_data = data[0:int(.2 * len(data))]
    training_data_master = data[int(.2 * len(data)):len(data)]

    # Shuffle full training data set.
    shuffled_training_data = training_data_master.sample(frac=1)

    # Create K random divisions of the test data and store them in a pandas dataframe.
    K_folds = pd.DataFrame(np.array_split(shuffled_training_data, K))

    ####################################################################################################################
    ############################################ BRAF Algorithm. #######################################################
    ####################################################################################################################

    ################################### Training Data K-fold Cross Validation. #########################################

    # Create empty data structures to hold the precision and recall values at the random forest level (compare outputs
    # every random forest to the predicted value).
    metrics_dict = {'precision': [], 'recall': []}

    # Create empty data structure to hold the ratio of trees that correctly predicted the training data outcomes in the
    # random forest.
    metrics_dict_tree_master = {'training_outcomes': [], 'probabilities': []}

    # for i in range(0, len(K_folds[0])):
    for i in range(0, 1):

        # Remove the first 1/10 of the data from the training dataset to be used as the testing data in each iteration.
        training_data_minus_fold = training_data_master.drop(K_folds[0][i].index)

        # Run BRAF algorithm to build model and calculate metrics from model using the 9/10ths of the data to train the
        # model and the remaining 1/10th to test the model.
        run_metrics = braf_main.braf(training_data=training_data_minus_fold, test_data=K_folds[0][i], s=s, p=p, K=K)

        # Attach values of precision and recall, training outcomes and probabilities for each iteration of the k-fold
        # cross validation.
        metrics_dict = dict_list_appender(metrics_dict, run_metrics[:-1])
        for key in metrics_dict_tree_master.keys():
            metrics_dict_tree_master[key] = metrics_dict_tree_master[key] + run_metrics[-2][key]

    # Extract values for false positive rate (fpr), true positive rate(tpr), precision, and area under curves for the
    # decision trees generated from the training data and plot the associated prc and roc curves.
    training_data_curves = curve_generator(*prc_roc_curve(
        np.array(metrics_dict_tree_master['training_outcomes']), np.array(metrics_dict_tree_master['probabilities'])))
    training_data_curves.gen_roc(title='Training Data')
    training_data_curves.gen_prc(title='Training Data')

    # Print out relevant run metrics for training data.
    print(f"Training Data AUROC = {training_data_curves.get_auc_roc()}")
    print(f"Training Data AUPRC = {training_data_curves.get_auc_prc()}")
    print(f"Training data metrics per run of k-fold cross validation (AVG) = {pd.DataFrame(metrics_dict)}")

    ################################ Execution of Best Model on Testing Data. ##########################################

    # Create empty data structures to hold the precision and recall values at the random forest level (compare outputs
    # every random forest to the predicted value).
    metrics_dict_test = {'precision': [], 'recall': []}

    # Create empty data structure to hold the ratio of trees that correctly predicted the training data outcomes in the
    # random forest.
    metrics_dict_tree_master_test = {'training_outcomes': [], 'probabilities': []}

    # Get Index of best model from training data (one with highest recall)
    best_index = metrics_dict['recall'].index(max(metrics_dict['recall']))

    # Train model on best k-fold of training data, and apply it to make predictions on test data.
    # # Calculate metrics from model.
    training_data_minus_fold_for_test = training_data_master.drop(K_folds[0][best_index].index)
    run_metrics_test = braf_main.braf(training_data=training_data_minus_fold_for_test, test_data=holdout_data,
                                      s=s, p=p, K=K)

    metrics_dict_test = dict_list_appender(metrics_dict_test, run_metrics_test[:-1])
    for key in metrics_dict_tree_master_test.keys():
        metrics_dict_tree_master_test[key] = metrics_dict_tree_master_test[key] + run_metrics_test[-2][key]

    # Extract values for false positive rate (fpr), true positive rate(tpr), precision, and area under curves for the
    # decision trees generated from the test data and plot the associated prc and roc curves.
    testing_data_curves = curve_generator(*prc_roc_curve(np.array(metrics_dict_tree_master_test['training_outcomes']),
                                                         np.array(metrics_dict_tree_master_test['probabilities'])))
    testing_data_curves.gen_roc(title='Testing Data')
    testing_data_curves.gen_prc(title='Testing Data')

    # Print out relevant run metrics for testing data.
    print(f"Testing Data AUROC = {testing_data_curves.get_auc_roc()}")
    print(f"Testing Data AUPRC = {testing_data_curves.get_auc_prc()}")
    print(f"Testing data metrics (AVG) = {metrics_dict_test}")


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)
