# -*- coding: utf-8 -*-
""" This module contains helper functions for calculating key metrics involved in model performance, namely, counts for
the precision, recall, and false positive rate(FPR).
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
    if false_positive + true_negative == 0:
        false_positive_rate = 0
    else:
        false_positive_rate = false_positive / (false_positive + true_negative)

    return precision, recall, false_positive_rate


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


def roc_curve(y, prob):
    tpr_list = []
    fpr_list = []
    threshold = np.linspace(1.1, 0, 10)
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
        auc = integrate(fpr_list, tpr_list)
    return fpr_list, tpr_list, threshold, auc


def plot_prc_curve(precision, recall, plot_title):
    # recall = np.linspace(0.0, 1.0, num=42)
    # precision = np.random.rand(42) * (1. - recall)

    # take a running maximum over the reversed vector of precision values, reverse the
    # result to match the order of the recall vector
    decreasing_max_precision = np.maximum.accumulate(precision[::-1])[::-1]
    fig, ax = plt.subplots(1, 1)
    ax.hold(True)
    ax.plot(recall, precision, '--b')
    ax.step(recall, decreasing_max_precision, '-r')
    # ax.legend()
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    plt.title(f'Precision-Recall Curve for {plot_title}')
    plt.show()
