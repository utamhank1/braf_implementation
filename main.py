import argparse
import os
import sys
import pandas as pd
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

    # Data Importation.
    raw_data = pd.DataFrame(pd.read_csv(file))
    # print(raw_data.head())

    # Data Exploration
    # Draw histograms of distributions of features for people with and without diabetes.
    # data_explorer.data_summary_statistics(raw_data).draw_distributions()
    # plt.show()

    # Draw correlational heatmap for all features.
    plt.figure(figsize=(10, 10))
    plt.show(data_explorer.data_summary_statistics(raw_data).draw_correlations())

    # Data preprocessing, three processed data objects created to test each imputation method.
    # processed_data_median_impute = data_preprocessor.preprocessed_data(raw_data, imputation_method='median')
    # processed_data_mean_impute = data_preprocessor.preprocessed_data(raw_data, imputation_method='mean')
    # processed_data_mean_sd_random_impute = data_preprocessor.preprocessed_data(raw_data, imputation_method='mean')


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
