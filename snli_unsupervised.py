import argparse
import os

import numpy as np
from sklearn.metrics import auc, roc_curve
from transformers.data.metrics import simple_accuracy, acc_and_f1


def two_classes(preds, out_label_ids):
    out_label_ids = np.where(out_label_ids == 1, 1, 0)
    best_result = None
    best_performance = -10
    epsilon = -1
    for e in np.arange(-1, 1.05, 0.05):
        e = round(e, 2)
        pred_label_ids = np.where(preds > e, 1, 0)
        tmp_result = acc_and_f1(pred_label_ids, out_label_ids)
        if tmp_result['acc_and_f1'] > best_performance:
            best_performance = tmp_result['acc_and_f1']
            best_result = tmp_result
            epsilon = e

    fpr, tpr, thresholds = roc_curve(out_label_ids, preds, pos_label=1)
    auc_result = auc(fpr, tpr)
    return epsilon, best_result, auc_result


def two_classes_fixed(preds, out_label_ids, e):
    out_label_ids = np.where(out_label_ids == 1, 1, 0)
    best_result = None
    best_performance = -10
    epsilon = -1

    pred_label_ids = np.where(preds > e, 1, 0)
    tmp_result = acc_and_f1(pred_label_ids, out_label_ids)
    if tmp_result['acc_and_f1'] > best_performance:
        best_performance = tmp_result['acc_and_f1']
        best_result = tmp_result
        epsilon = e

    fpr, tpr, thresholds = roc_curve(out_label_ids, preds, pos_label=1)
    auc_result = auc(fpr, tpr)
    return epsilon, best_result, auc_result


# def three_classes(epsilon, preds, out_label_ids):
#     best_result = None
#     best_performance = -10
#     alpha = -1
#     for a in np.arange(-1, epsilon, 0.05):
#         a = round(a, 2)
#         pred_label_ids = np.where(preds > epsilon, 1,
#                                   np.where(preds < a, 0, 2))
#         tmp_result = simple_accuracy(pred_label_ids, out_label_ids)
#         if tmp_result > best_performance:
#             best_performance = tmp_result
#             best_result = tmp_result
#             alpha = a
#     return alpha, best_result


def three_classes(preds, out_label_ids):
    best_result = None
    best_performance = -10
    # alpha = -1
    a1, a2 = -1, -1
    for psi_1 in np.arange(-1, 1.05, 0.05):
        psi_1 = round(psi_1, 2)
        for psi_2 in np.arange(psi_1, 1.05, 0.05):
            if psi_1 == psi_2:
                continue
            pred_label_ids = np.where(preds > psi_2, 1,
                                      np.where(preds < psi_1, 0, 2))
            tmp_result = simple_accuracy(pred_label_ids, out_label_ids)
            if tmp_result > best_performance:
                best_performance = tmp_result
                best_result = tmp_result
                a1 = psi_1
                a2 = psi_2
    return a1, a2, best_result


def three_classes_fixed(epsilon, preds, out_label_ids, a):
    best_result = None
    best_performance = -10

    pred_label_ids = np.where(preds > epsilon, 1,
                              np.where(preds < a, 0, 2))
    tmp_result = simple_accuracy(pred_label_ids, out_label_ids)
    if tmp_result > best_performance:
        best_performance = tmp_result
        best_result = tmp_result
        alpha = a
    return a, best_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str)

    args = parser.parse_args()

    data_folder = args.data_folder
    preds = np.load(os.path.join(data_folder, 'preds.npy'))
    out_label_ids = np.load(os.path.join(data_folder, 'out_label_ids.npy'))

    e, two_class_result, auc_result = two_classes(preds, out_label_ids)
    # e, two_class_result, auc_result = two_classes_fixed(preds, out_label_ids, 0.75)
    print('Two Classes:')
    print('Epsilon', e)
    print(two_class_result)
    print("AUC", auc_result)

    a1, a2, three_class_result = three_classes(preds, out_label_ids)
    # a, three_class_result = three_classes_fixed(e, preds, out_label_ids, 0.45)
    print('Three Classes:')
    print("Alpha1", a1, 'Alpha2', a2)
    print(three_class_result)