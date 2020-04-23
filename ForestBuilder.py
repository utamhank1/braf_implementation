import numpy as np
import math


def bagger(dataset, k):
    """
    Builds subsample of a larger dataset by picking k rows with replacement.
    :param dataset: numpy array
    :param k: User set condition for k indices to randomly select
    :return: numpy array of the k indexes (seeds)
    """
    k_indices = np.random.choice(len(dataset), k)
    return dataset[k_indices]


def calc_information_gain(groups, list_of_class_ids):
    # count all samples
    Nall = sum([len(group) for group in groups])
    # calculate Gini index of parent node
    all_rows = [row for group in groups for row in group]
    IG = calc_gini(all_rows, list_of_class_ids)
    # calculate Gini index of daughter nodes
    for group in groups:
        IG -= calc_gini(group, list_of_class_ids) * len(group) / Nall
    return IG


def calc_gini(group, list_of_class_ids):
    Ngroup = len(group)
    if Ngroup == 0:
        return 0
    dataset_class_ids = [row[-1] for row in group]
    sum_over_classes = 0.
    for class_id in list_of_class_ids:
        prob = dataset_class_ids.count(class_id) / Ngroup
        sum_over_classes += prob ** 2
    return 1. - sum_over_classes


def split_node(index, value, dataset):
    ''' Split the dataset into two using a feature index and
    feature value '''
    left = []
    right = []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return [left, right]


def get_split(dataset, index):
    ''' Evaluate all possible splits given the dataset and the index of
    the feature on which to split '''
    list_of_class_ids = list(set(row[-1] for row in dataset))
    split_value, max_IG, split_groups = 0., -1., None
    for row in dataset:
        groups = split_node(index, row[index], dataset)
        IG = calc_information_gain(groups, list_of_class_ids)
        if IG > max_IG:
            split_value, max_IG, split_groups = row[index], IG, groups
    return {'index': index, 'split_value': split_value, 'groups': groups}


def build_tree(train, max_depth, min_size):
    # randomly determine the feature index
    feature_index = int(np.random.random() * (len(train[0]) - 1))
    root = get_split(train, feature_index)
    split(root, max_depth, min_size, 1)
    return root


def to_terminal(group):
    # Create a terminal node value
    list_of_classes = [row[-1] for row in group]
    return max(set(list_of_classes), key=list_of_classes.count)


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'] = to_terminal(left)
        node['right'] = to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        feature_index = int(np.random.random() * (len(right[0]) - 1))
        node['left'] = get_split(left, feature_index)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        feature_index = int(np.random.random() * (len(right[0]) - 1))
        node['right'] = get_split(right, feature_index)
        split(node['right'], max_depth, min_size, depth + 1)


def build_forest(train, k, N_trees):
    max_depth = math.sqrt(len(train[0]))
    min_size = 2
    forest = []
    for i in range(0, int(N_trees)):
        # bootstrap training dataset
        k_indices = np.random.choice(len(train), k)
        forest.append(build_tree(train[k_indices], max_depth, min_size))
    return forest


def traverse_tree(node, row):
    if row[node['index']] < node['split_value']:
        if isinstance(node['left'], dict):
            return traverse_tree(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return traverse_tree(node['right'], row)
        else:
            return node['right']


def make_prediction(forest, row):
    list_of_classes = []
    for tree_root in forest:
        list_of_classes.append(traverse_tree(tree_root, row))
    # print(f"list_of_classes={list_of_classes}")
    # print(f"list_of_classes.count = {list_of_classes.count}")
    # print(f"max(set(list_of_classes), key=list_of_classes.count) = {max(set(list_of_classes), key=list_of_classes.count)}")
    # print(f"set(list_of_classes)={set(list_of_classes)}")
    return max(set(list_of_classes), key=list_of_classes.count)
