import torch
from tqdm import tqdm
from utils.confusion_matrix import ConfusionMatrix


class Evaluator:

    def __init__(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                 device: torch.device, use_depth: bool):
        self.model = model
        self.num_semantic_classes = self.model.n_classes
        self.num_inputs = self.model.n_inputs
        self.device = device

        self.loader = data_loader
        self.data_len = data_loader.__len__()
        self.use_depth = use_depth
        self.semantic_conf_matrix = ConfusionMatrix(self.device, self.num_semantic_classes)

    def evaluate(self):
        self._clear_cache()
        self.model.eval()
        with torch.no_grad():
            for index, sample in tqdm(enumerate(self.loader)):
                image, depth, semantic = sample
                semantic = semantic.to(self.device)
                # Do predictions.
                if self.use_depth:
                    semantic_prediction = self.model(depth.to(self.device))
                else:
                    semantic_prediction = self.model(image.to(self.device))
                # Evaluate semantic segmentation.
                for i in range(image.shape[0]):
                    self._evaluate_semantic(semantic_prediction[i], semantic[i])
        # Retrieve metrics.
        miou = self._compute_semantic_miou()
        fw_miou = self._compute_semantic_miou(frequency_weighted=True)
        pixel_acc = self._compute_semantic_pixel_accuracy()
        class_acc = self._compute_semantic_pixel_accuracy(per_class=True)
        return miou, fw_miou, pixel_acc, class_acc

    def _evaluate_semantic(self, pred, gt):
        # Get the max prediction, which is the integer of the class.
        _, pred_label = torch.max(pred, dim=0)
        # Get the confusion matrix for the IoU.
        self.semantic_conf_matrix.add_to_matrix(pred_label, gt)

    def _compute_semantic_miou(self, frequency_weighted=False):
        # Get the TP, FP, FNs.
        intersection = torch.diag(self.semantic_conf_matrix.matrix)
        ground_truth_set = torch.sum(self.semantic_conf_matrix.matrix, dim=1)
        predicted_set = torch.sum(self.semantic_conf_matrix.matrix, dim=0)
        # The class IoU is TP / (FP + FN + TP)
        # = intersection / ((gt_set - intersection) + (p_set - intersection) + intersection)
        # = intersection / ground_truth_set + predicted_set - intersection
        # = intersection / union
        union = ground_truth_set + predicted_set - intersection
        # Assuming all classes are present in the ground-truth, else we get nasty zero-divisions.
        class_iou = intersection / union
        if not frequency_weighted:
            return torch.nanmean(class_iou).item()
        # Frequency weighted mIoU are also required for evaluation.
        freq = self.semantic_conf_matrix.matrix.sum(axis=1) / self.semantic_conf_matrix.matrix.sum()
        return (freq[freq > 0] * class_iou[freq > 0]).sum().item()

    def _compute_semantic_pixel_accuracy(self, per_class=False):
        # Per pixel accuracy.
        if not per_class:
            return (torch.diag(self.semantic_conf_matrix.matrix).sum() / self.semantic_conf_matrix.matrix.sum()).item()
        # Per class accuracy.
        class_acc = torch.diag(self.semantic_conf_matrix.matrix) / self.semantic_conf_matrix.matrix.sum(axis=1)
        return torch.nanmean(class_acc).item()

    def _clear_cache(self):
        self.semantic_conf_matrix = ConfusionMatrix(self.device, self.num_semantic_classes)
