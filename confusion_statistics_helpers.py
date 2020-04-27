# -*- coding: utf-8 -*-
""" This module contains helper functions for calculating key metrics involved in model performance, namely, counts for
the precision, recall, and false positive rate(FPR) as well as the associated graphs.
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb


def dict_list_appender(dictionary1, list1):
    for key, index in zip(dictionary1.keys(), range(0, len(list1))):
        dictionary1[key].append(list1[index])
    return dictionary1


def confusion_calculator(prediction_list, value):
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
    if value == 1:
        prob_hit = sum(prediction_list) / len(prediction_list)
    else:
        prob_hit = (len(prediction_list) - sum(prediction_list)) / len(prediction_list)
    return value, prob_hit


def integrate(x, y):
    sm = 0
    for i in range(1, len(x)):
        h = x[i] - x[i - 1]
        sm += h * (y[i - 1] + y[i]) / 2

    return sm


def prc_roc_curve(y, prob):
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
        auc_prc = integrate(tpr_list, precision_list)
        auc_roc = integrate(fpr_list, tpr_list)
    return fpr_list, tpr_list, precision_list, auc_roc, auc_prc


class curve_generator(object):

    def __init__(self, fpr, tpr, precision, auc_roc, auc_prc):
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
        plt.plot(self.get_fpr(), self.get_tpr(), 'b')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title(f"{title} Performance ROC: AUC = {self.get_auc_roc()}")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.show()

    def gen_prc(self, title):
        plt.plot(self.get_tpr(), self.get_precision(), 'b')
        plt.plot([0, 1], [.8, .8], 'r--')
        plt.title(f"{title} Performance PRC: AUC = {self.get_auc_prc()}")
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)

        plt.show()








    # recall = np.linspace(0.0, 1.0, num=42)
    # precision = np.random.rand(42) * (1. - recall)

    # take a running maximum over the reversed vector of precision values, reverse the
    # result to match the order of the recall vector
    # decreasing_max_precision = np.maximum.accumulate(precision[::-1])[::-1]
    # fig, ax = plt.subplots(1, 1)
    # ax.hold(True)
    # ax.plot(recall, precision, '--b')
    # ax.step(recall, decreasing_max_precision, '-r')
    # # ax.legend()
    # ax.set_xlabel("recall")
    # ax.set_ylabel("precision")
    # plt.title(f'Precision-Recall Curve for {plot_title}')
    # plt.show()
