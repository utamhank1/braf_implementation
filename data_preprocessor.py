from scipy import stats
import numpy as np


class preprocessed_data(object):

    def __init__(self, raw_data, imputation_method='mean', stdev_to_keep=2.75):
        self.raw_data = raw_data
        self.imputation_method = imputation_method
        self.stdev_to_keep = stdev_to_keep

    def get_imputation_method(self):
        return self.imputation_method

    def get_raw_data(self):
        return self.raw_data

    def get_stand_dev_to_keep(self):
        return self.stdev_to_keep

    def outlier_removed_data(self):
        # Remove nan values
        df_without_na = self.get_raw_data().fillna(0)
        # Remove outlier values by checking each column, and removing the row where the value for that column is greater
        # than the supplied standard deviation number
        return df_without_na[(stats.zscore(df_without_na) < self.get_stand_dev_to_keep()).all(axis=1)]

    def impute(self):
        df_original = self.outlier_removed_data()
        df_out = self.outlier_removed_data()
        imputation_method = self.get_imputation_method()
        columns_2_thru_8 = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                            'BMI', 'DiabetesPedigreeFunction', 'Age']

        if imputation_method == 'mean':
            df_out[columns_2_thru_8] = df_out[columns_2_thru_8].replace(0, df_out.mean())
            return df_out

        if imputation_method == 'median':
            df_out[columns_2_thru_8] = df_out[columns_2_thru_8].replace(0, df_out.median())
            return df_out

        if imputation_method == 'random':
            df_out[columns_2_thru_8] = df_out[columns_2_thru_8].replace(0, df_out.mean())
            df_original = df_original.reset_index(drop=True)
            data_summary = df_out.describe()

            for i in range(1, len(df_original.columns) - 1):
                mean = data_summary.iloc[:, i]['mean']
                std_dev = data_summary.iloc[:, i]['std']
                for j in range(0, len(df_original.iloc[:, i])):
                    if df_original.iloc[:, i][j] < .0001:
                        df_original.iloc[:, i][j] = abs(np.random.normal(mean, std_dev))
            return df_original
