# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 18:21:15 2023

@author: giosp
"""
import numpy as np

def show_metrics(y_true, y_prediction):
    # True positive
    tp = np.sum((y_true==1) & (y_prediction==1))
    # False positive
    fp = np.sum((y_true==0) & (y_prediction == 1))
    # True negative
    tn = np.sum((y_true==0) & (y_prediction==0))
    # False negative
    fn = np.sum((y_true==1) & (y_prediction==0))

    # True positive rate (sensitivity or recall)
    tpr = tp / (tp + fn)
    # False positive rate (fall-out)
    fpr = fp / (fp + tn)
    # Precision
    precision = tp / (tp + fp)
    # True negatvie rate (specificity)
    tnr = 1 - fpr
    # F1 score
    f1 = 2*tp / (2*tp + fp + fn)
    # ROC-AUC for binary classification
    auc = (tpr+tnr) / 2
    # MCC
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    print("True positive: ", tp)
    print("False positive: ", fp)
    print("True negative: ", tn)
    print("False negative: ", fn)

    print("True positive rate (recall): ", tpr)
    print("False positive rate: ", fpr)
    print("Precision: ", precision)
    print("True negative rate: ", tnr)
    print("F1: ", f1)
    print("ROC-AUC: ", auc)
    print("MCC: ", mcc)