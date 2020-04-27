# -*- coding: utf-8 -*-
""" This module contains objects and basic visualization tools for preprocessing the Pima diabetes dataset.
The preprocessed_data
"""

from scipy import stats
import numpy as np
import seaborn as sns
import collections
import matplotlib.pyplot as plt


def gen_preprocessed_objects(imputation_methods, std_dev_to_keep, raw_data):
    # # Create preprocessed and imputed data objects with varying std. deviations kept and imputation methods.
    processed_data_objects = collections.defaultdict(list)
    for imputation_method in imputation_methods:
        for std_dev in std_dev_to_keep:
            processed_data_objects[("data_std_dev_" + str(std_dev).replace('.', '_') + "_impute_" +
                                    str(imputation_method))].append(preprocessed_data(raw_data, stdev_to_keep=std_dev).
                                                                    impute(imputation_method=imputation_method))
    return processed_data_objects


class preprocessed_data(object):

    def __init__(self, raw_data, stdev_to_keep=2.75):
        self.raw_data = raw_data
        self.stdev_to_keep = stdev_to_keep

    def get_raw_data(self):
        return self.raw_data

    def get_std_dev_to_keep(self):
        return self.stdev_to_keep

    def outlier_removed_data(self):
        """
        Function that removes nan values and outliers from the data based on a user supplied value for "stddev_to_keep"
        which refers to the std deviations of the data to include in this functions output.
        :return: pandas dataframe
        """
        # Remove nan values
        df_without_na = self.get_raw_data().fillna(0)

        # Remove outlier values by checking each column, and removing the row where the value for that column is greater
        # than the supplied standard deviation number
        return df_without_na[(stats.zscore(df_without_na) < self.get_std_dev_to_keep()).all(axis=1)]

    def impute(self, imputation_method='mean'):
        """
        Impute missing values for the data based on the imputation method called by the user.
        :param imputation_method: string
        :return: pandas dataframe with data imputation performed.
        """

        df_original = self.outlier_removed_data()
        df_out = self.outlier_removed_data()
        columns_2_thru_8 = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                            'BMI', 'DiabetesPedigreeFunction', 'Age']

        if imputation_method == 'mean':
            df_out[columns_2_thru_8] = df_out[columns_2_thru_8].replace(0, df_out.mean())
            return df_out.reset_index(drop=True)

        if imputation_method == 'median':
            df_out[columns_2_thru_8] = df_out[columns_2_thru_8].replace(0, df_out.median())
            return df_out.reset_index(drop=True)

        if imputation_method == 'random':

            # Create a copy of the original dataframe and replace the missing values in the dataframe with
            # the mean value of the missing values' column.
            df_out[columns_2_thru_8] = df_out[columns_2_thru_8].replace(0, df_out.mean())

            # Create a copy of the original dataframe and reset its index to account for any outlier values removed as
            # a result of the outlier_removed_data() function, but don't replace the missing values in this copy.
            df_original = df_original.reset_index(drop=True)

            # Create a new dataframe that holds summary statistics of the the copied dataframe.
            data_summary = df_out.describe()

            # For each value of 0 in columns 2 through 8, replace the missing values with random numbers sampled from a
            # Gaussian distribution with parameters for the mean and std dev based on the specific column that the
            # missing value is located in.
            for i in range(1, len(df_original.columns) - 1):

                mean = data_summary.iloc[:, i]['mean']
                std_dev = data_summary.iloc[:, i]['std']

                for j in range(0, len(df_original.iloc[:, i])):

                    if df_original.iloc[:, i][j] < .0001:
                        # Using the summary statistics for the dataframe with the zeros replaced by the mean
                        # values (data_summary) to create normal distributions from which we sample random values to
                        # insert into "df_original" (which does not have zeros removed).
                        df_original.iloc[:, i][j] = abs(np.random.normal(mean, std_dev))

            return df_original


class data_explorer(preprocessed_data):

    def __init__(self, raw_data):
        super().__init__(raw_data)

    def draw_distributions(self):
        data = self.get_raw_data()
        # Save histograms for both distributions of each feature for Outcome=1 and Outcome=0.
        for i in data['Outcome'].unique():
            data.loc[data['Outcome'] == i].hist(figsize=(9, 9))
            plt.savefig(f'Outcome_{i}_histograms.png')
            plt.clf()
        return None

    def draw_correlations(self):
        correlations = abs(self.get_raw_data().corr())
        sns.set(font_scale=.5)
        return sns.heatmap(correlations, annot=True, robust=True, linecolor='white', cbar=False, cmap="YlGnBu")

    def print_summary(self):
        return self.get_raw_data().describe()
