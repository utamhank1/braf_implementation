import numpy as np


def bagger(dataset, k):
    """
    Builds subsample of a larger dataset by picking k rows with replacement.
    :param dataset: numpy array
    :param k: User set condition for k indices to randomly select
    :return: numpy array of the k indexes (seeds)
    """
    k_indices = np.random.choice(len(dataset), k)
    return dataset[k_indices]

