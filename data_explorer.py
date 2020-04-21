import seaborn as sns


class data_summary_statistics:

    def __init__(self, raw_data):
        self.raw_data = raw_data

    def get_raw_data(self):
        return self.raw_data

    def draw_distributions(self):
        self.get_raw_data().groupby('Outcome').hist(figsize=(9, 9))

    def draw_correlations(self):
        correlations = abs(self.get_raw_data().corr())
        sns.set(font_scale=.5)
        return sns.heatmap(correlations, annot=True, robust=True, linecolor='white', cbar=False, cmap="YlGnBu")

    def print_summary(self):
        return self.get_raw_data().describe()


