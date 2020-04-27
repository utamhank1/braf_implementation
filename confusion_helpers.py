# -*- coding: utf-8 -*-
""" This module contains helper functions for calculating key metrics involved in model performance, namely, counts for
the precision, recall as well as creating and saving associated ROC and PRC graphs.
"""
import numpy as np
import matplotlib.pyplot as plt


def dict_list_appender(dictionary1, list1):
    """
    This function takes a dictionary with values that are lists and appends the elements of the supplied list to every
    value in the list.
    :param dictionary1: dict with key-value pair format {key1: list, key2: list2 ... }
    :param list1: list.
    :return: dictionary with appended values to every key-value pair.
    """
    for key, index in zip(dictionary1.keys(), range(0, len(list1))):
        dictionary1[key].append(list1[index])
    return dictionary1


def confusion_calculator(prediction_list, value):
    """
    Calculates the "confusion matrix" of a prediction list and a value by returning the precision and recall
    :param prediction_list: list of binary integers.
    :param value: int binary 0 or 1.
    :return: float, float precision and recall values.
    """
    true_positive = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negative = 0.0
    for item in prediction_list:
        if item != value:
            if item == 1 and value == 0:
                false_positive += 1
            else:
                false_negative += 1
        elif item == 0:
            true_negative += 1
        else:
            true_positive += 1
    precision = true_positive / len(prediction_list)
    # Handle edge cases
    if true_positive + false_negative == 0:
        recall = 1
    else:
        recall = true_positive / (true_positive + false_negative)

    return precision, recall


def tree_probability_calculator(prediction_list, value):
    """
    Calculates the probabilities of every item in the supplied random forest prediction list
    for the possible values 0 and 1.
    :param prediction_list: list of binary integers.
    :param value: int binary 0 or 1.
    :return: int, float representing the probability of getting that particular value while traversing the tree(s).
    """
    if value == 1:
        prob_hit = sum(prediction_list) / len(prediction_list)
    else:
        prob_hit = (len(prediction_list) - sum(prediction_list)) / len(prediction_list)
    return value, prob_hit


def integrate(x, y):
    """
    Calculates the area inder a curve.
    :param x: list of floats or int.
    :param y: list of floats or int.
    :return: float.
    """
    sm = 0
    for i in range(1, len(x)):
        h = x[i] - x[i - 1]
        sm += h * (y[i - 1] + y[i]) / 2

    return sm


def prc_roc_curve(y, prob):
    """
    This function calculates the list of false positive rates, true positive rates, precisions and area under the
    ROC and PRC curves associated with ta list of true outcomes and their associated probabilities from the model.
    :param y: list of binary values.
    :param prob: list of floats representing probabilities.
    :return:
    """
    tpr_list = []
    fpr_list = []
    precision_list = []
    threshold = np.linspace(1.1, 0, len(prob))
    for t in threshold:
        y_pred = np.zeros(y.shape[0])
        y_pred[prob >= t] = 1
        TN = y_pred[(y_pred == y) & (y == 0)].shape[0]
        TP = y_pred[(y_pred == y) & (y == 1)].shape[0]
        FP = y_pred[(y_pred != y) & (y == 0)].shape[0]
        FN = y_pred[(y_pred != y) & (y == 1)].shape[0]
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        tpr_list.append(TPR)
        fpr_list.append(FPR)
        if TPR + FPR == 0:
            precision_list.append(1)
        else:
            precision_list.append(TPR / (FPR + TPR))
        # Determine area under the curve.
        auc_prc = integrate(tpr_list, precision_list)
        auc_roc = integrate(fpr_list, tpr_list)
    return fpr_list, tpr_list, precision_list, auc_roc, auc_prc


class curve_generator(object):

    def __init__(self, fpr, tpr, precision, auc_roc, auc_prc):
        """
        This object uses the supplied values of fpr, tpr, precision, auc_roc, and auc_prc to generate the associated
        ROC and PRC plots and save them to disk.
        :param fpr: list of floats of false positive rates.
        :param tpr: list of floats of true positive rates.
        :param precision: list of floats of precisions.
        :param auc_roc: float area under roc curve value.
        :param auc_prc: float area under curve of prc curve.
        """
        self.fpr = fpr
        self.tpr = tpr
        self.precision = precision
        self.auc_roc = auc_roc
        self.auc_prc = auc_prc

    def get_tpr(self):
        return self.tpr

    def get_fpr(self):
        return self.fpr

    def get_precision(self):
        return self.precision

    def get_auc_roc(self):
        return self.auc_roc

    def get_auc_prc(self):
        return self.auc_prc

    def gen_roc(self, title):
        """
        Plot the ROC curve and save it to file.
        :param title: string representing the type of data this graph is for (training/testing).
        :return: None.
        """
        plt.plot(self.get_fpr(), self.get_tpr(), 'b')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title(f"{title} Performance ROC: AUC = {self.get_auc_roc()}")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.savefig(f"ROC_{title}.png")
        plt.clf()

    def gen_prc(self, title):
        """
        Plot the PRC curve and save it to file.
        :param title: string representing the type of data this graph is for (training/testing).
        :return: None.
        """

        plt.plot(self.get_tpr(), self.get_precision(), 'b')
        plt.plot([0, 1], [.8, .8], 'r--')
        plt.title(f"{title} Performance PRC: AUC = {self.get_auc_prc()}")
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.savefig(f"PRC_{title}.png")
        plt.clf()
