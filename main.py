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
    ############################################ Data Exploration ######################################################
    ####################################################################################################################

    # Draw correlation heatmap for all features.
    plt.figure(figsize=(10, 10))
    plt.show(data_preprocessor.data_explorer(raw_data).draw_correlations())

    # Draw histograms of distributions of features for people with and without diabetes.
    data_preprocessor.data_explorer(raw_data).draw_distributions()
    plt.show()

    # Save summary statistics for each feature.
    feature_summary_statistics = data_preprocessor.data_explorer(raw_data).print_summary()

    ####################################################################################################################
    ############################################ Data Pre-processing####################################################
    ####################################################################################################################
    print(data_preprocessor.preprocessed_data(raw_data, stdev_to_keep=2.75).impute(imputation_method='random'))


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
