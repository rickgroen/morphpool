# bfscore: Contour/Boundary matching score for multi-class image segmentation #
# Reference: Csurka, G., D. Larlus, and F. Perronnin. "What is a good evaluation measure for semantic segmentation?" Proceedings of the British Machine Vision Conference, 2013, pp. 32.1-32.11. #
# Crosscheck: https://www.mathworks.com/help/images/ref/bfscore.html #
import os.path

import cv2
import numpy as np
import math


def calc_precision_recall(contours_a, contours_b, threshold):
    """ For precision, contours_a==GT & contours_b==Prediction
        For recall, contours_a==Prediction & contours_b==GT
    """
    x = contours_a
    y = contours_b
    xx = np.array(x)
    hits = []
    for yrec in y:
        d = np.square(xx[:,0] - yrec[0]) + np.square(xx[:,1] - yrec[1])
        hits.append(np.any(d < threshold*threshold))
    top_count = np.sum(hits)
    try:
        precision_recall = top_count / len(y)
    except ZeroDivisionError:
        precision_recall = 0
    return precision_recall, top_count, len(y)


def compute_bfscore_batch_with_load(job):
    gt_path, pr_path = job
    gt = np.load(gt_path)
    pr = np.load(pr_path)
    bf_score = compute_bfscore(gt, pr)
    os.unlink(gt_path)
    os.unlink(pr_path)
    return bf_score


def compute_bfscore(gt_, pr_):
    """ computes the BF (Boundary F1) contour matching score between the predicted and GT segmentation
        from: https://github.com/minar09/bfscore_python/blob/master/bfscore.py
    """
    # Compute threshold as 0.75% of the image diagonal.
    image_diagonal = np.sqrt(gt_.shape[0] ** 2 + gt_.shape[1] ** 2)
    # Get an odd-valued pixel error for an odd number of pixels.
    threshold = math.ceil(image_diagonal * 0.0075)

    classes_gt = np.unique(gt_)    # Get GT classes
    classes_pr = np.unique(pr_)    # Get predicted classes
    # Check classes from GT and prediction
    if not np.array_equiv(classes_gt, classes_pr):
        classes = np.concatenate((classes_gt, classes_pr))
        classes = np.unique(classes)
        classes = np.sort(classes)
    else:
        classes = classes_gt    # Get matched classes

    m = np.max(classes)    # Get max of classes (number of classes)
    # Define bfscore variable (initialized with zeros)
    bfscores = np.zeros(m, dtype=float)
    for i in range(m):
        bfscores[i] = np.nan

    for target_class in classes:    # Iterate over classes
        if target_class == 0:     # Skip background
            continue

        gt = gt_.copy()
        gt[gt != target_class] = 0
        contours, _ = cv2.findContours(gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # Find contours of the shape

        # contours list of numpy arrays
        contours_gt = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours_gt.append(contours[i][j][0].tolist())
        if len(contours_gt) == 0:
            continue

        pr = pr_.copy()
        pr[pr != target_class] = 0
        contours, _ = cv2.findContours(pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # contours list of numpy arrays
        contours_pr = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours_pr.append(contours[i][j][0].tolist())
        if len(contours_pr) == 0:
            bfscores[target_class - 1] = 0.
            continue

        # 3. Calculate precision and recall.
        precision, _, _ = calc_precision_recall(contours_gt, contours_pr, threshold)    # Precision
        recall, _, _ = calc_precision_recall(contours_pr, contours_gt, threshold)    # Recall
        # 4. Compute the boundary F1-score.
        if np.abs(recall + precision) < 1e-4:
            # If recall and precision are zero, we would get a division error. However,
            # set that element to zero.
            f1 = 0.0
        else:
            # Else compute the boundary F1 score.
            f1 = 2 * recall * precision / (recall + precision)    # F1 score
        bfscores[target_class - 1] = f1
    return np.nanmean(bfscores)
