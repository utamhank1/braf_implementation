# -*- coding: utf-8 -*-
""" This module contains helper functions for calculating key metrics involved in model performance, namely, counts for
the precision, recall, and false positive rate(FPR).
"""


def dict_list_appender(dictionary1, list1):
    for key, index in zip(dictionary1.keys(), range(0, len(list1))):
        dictionary1[key].append(list1[index])
    return dictionary1


def confusion_calculator(prediction_list, value):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
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
    precision = true_positive / (true_positive + false_positive)
    # Handle edge cases
    if true_positive + false_negative != 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 1
    if false_positive + true_negative != 0:
        false_positive_rate = false_positive / (false_positive + true_negative)
    else:
        false_positive_rate = 0

    return precision, recall, false_positive_rate
