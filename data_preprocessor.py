from scipy import stats


class preprocessed_data:

    def __init__(self, raw_data, imputation_method='mean', stdev_to_keep=2.75):
        self.raw_data = raw_data
        self.imputation_method = imputation_method
        self.stdev_to_keep = stdev_to_keep

    def get_imputation_method(self):
        return self.imputation_method

    def get_raw_data(self):
        return self.raw_data

    def clean_data(self):
        # Remove nan values
        df_without_na = self.get_raw_data().fillna(0)
        # Remove outlier values by checking each column, and removing the row where the value for that column is greater
        # than the supplied standard deviation number
        return df_without_na[(stats.zscore(df_without_na) < self.stdev_to_keep).all(axis=1)]

    def impute(self):
        df_out = self.clean_data()
        imputation_method = self.get_imputation_method()

        if imputation_method == 'mean':

            df_clean = df_out[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                               'DiabetesPedigreeFunction', 'Age']].replace(0, df_out.mean())
            print('Executing...')
            df_out[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                    'DiabetesPedigreeFunction', 'Age']] = df_clean
            return df_out

        if imputation_method == 'median':
            df_clean = df_out[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                               'DiabetesPedigreeFunction', 'Age']].replace(0, df_out.median())
            df_out[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                    'DiabetesPedigreeFunction', 'Age']] = df_clean
            return df_out
