#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from sklearn.metrics.cluster import (contingency_matrix,
                                     normalized_mutual_info_score)
from sklearn.metrics import (precision_score, recall_score)

__all__ = ['bcubed', 'nmi', 'precision', 'recall', 'accuracy']


def _check(gt_labels, pred_labels):
    if gt_labels.ndim != 1:
        raise ValueError("gt_labels must be 1D: shape is %r" %
                         (gt_labels.shape, ))
    if pred_labels.ndim != 1:
        raise ValueError("pred_labels must be 1D: shape is %r" %
                         (pred_labels.shape, ))
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            "gt_labels and pred_labels must have same size, got %d and %d" %
            (gt_labels.shape[0], pred_labels.shape[0]))
    return gt_labels, pred_labels


def _get_lb2idxs(labels):
    lb2idxs = {}
    for idx, lb in enumerate(labels):
        if lb not in lb2idxs:
            lb2idxs[lb] = []
        lb2idxs[lb].append(idx)
    return lb2idxs


def _compute_fscore(pre, rec):
    return 2. * pre * rec / (pre + rec)


def bcubed(gt_labels, pred_labels):
    _check(gt_labels, pred_labels)

    gt_lb2idxs = _get_lb2idxs(gt_labels)
    pred_lb2idxs = _get_lb2idxs(pred_labels)

    num_lbs = len(gt_lb2idxs)
    pre = np.zeros(num_lbs)
    rec = np.zeros(num_lbs)
    gt_num = np.zeros(num_lbs)

    for i, gt_idxs in enumerate(gt_lb2idxs.values()):
        all_pred_lbs = np.unique(pred_labels[gt_idxs])
        gt_num[i] = len(gt_idxs)
        for pred_lb in all_pred_lbs:
            pred_idxs = pred_lb2idxs[pred_lb]
            n = 1. * np.intersect1d(gt_idxs, pred_idxs).size
            pre[i] += n**2 / len(pred_idxs)
            rec[i] += n**2 / gt_num[i]

    gt_num = gt_num.sum()
    avg_pre = pre.sum() / gt_num
    avg_rec = rec.sum() / gt_num
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore


def nmi(gt_labels, pred_labels):
    return normalized_mutual_info_score(pred_labels, gt_labels)


def precision(gt_labels, pred_labels):
    return precision_score(gt_labels, pred_labels)


def recall(gt_labels, pred_labels):
    return recall_score(gt_labels, pred_labels)


def accuracy(gt_labels, pred_labels):
    return np.mean(gt_labels == pred_labels)
