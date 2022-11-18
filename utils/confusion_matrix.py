import torch


class ConfusionMatrix:
    """
        Pytorch implementation of the confusion matrix to compute mIoU, so that we can
        put it on GPU and speed up validation set evaluation times.
    """

    def __init__(self, device, num_classes):
        self.device = device
        self.num_classes = num_classes

        # Make a float32 matrix to do additions to the confusion matrix.
        self.matrix = torch.zeros((num_classes, num_classes), requires_grad=False, dtype=torch.int64).to(self.device)
        # Ignore label -1
        self.ignore_label = -1

    def add_to_matrix(self, pred, gt):
        """ Adds the the ground truth and predictions to the matrix. Both inputs are expected to
            be [1, height, width]
        """
        # Flatten both tensors.
        flattened_pred = pred.view(-1)
        flattened_gt = gt.view(-1)
        # Stack tensors.
        stacked_pred_gt = torch.stack((flattened_gt, flattened_pred), dim=1)
        # Get the uniques, their counts and the indices in the confusion matrix.
        uniques, counts = torch.unique(stacked_pred_gt, return_counts=True, dim=0)
        # Remove the ignored labels.
        true_labels = torch.where(uniques[:, 0] > self.ignore_label)
        uniques = uniques[true_labels]
        counts = counts[true_labels]
        # Get the indices in the confusion matrix.
        indices = tuple(uniques.T)
        # Make a confusion matrix and add it to the global matrix.
        self.matrix[indices] += counts

    def _to_numpy(self):
        """ Mainly to check if the implementation is the same as scikit's.
        """
        return self.matrix.cpu().numpy()
