import numpy as np

class BuildTree(object):

    def __init__(self, data, depth, k):
        self.data = data
        self.depth = depth
        self.k = k

    def get_data(self):
        return self.data

    def get_depth(self):
        return self.depth

    def get_k(self):
        return self.k

    def get_indices(self):
        # Bootstrap a random amount of k indices
        k_indices = np.random_choice(len(self.get_data()), self.get_k())
        return self.get_data()[k_indices]


