from .base import MetricsCalculatorBase
from ..core import PipelineError
import torch
from sklearn.metrics import confusion_matrix

import numpy as np

class MetricsCalculatorSegmentation(MetricsCalculatorBase):
    def __init__(self, num_classes,border=0.5):
        self.num_classes = num_classes
        self.border = border
        self.zero_cache()
        
    def zero_cache(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
    def add(self, y_predicted, y_true):
        
        # Convert tensor to numpy array
        if isinstance(y_predicted, torch.Tensor):
            y_predicted = y_predicted.cpu().data.numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().data.numpy()
        assert isinstance(y_true, np.ndarray) and isinstance(y_predicted, np.ndarray)
        # Binarize the prediction
        if self.num_classes == 1:
            # Binary classification
            y_predicted = (y_predicted >= self._border).astype("int")
        else:
            y_predicted = np.argmax(y_predicted, -1)
            y_true = np.argmax(y_true, -1)
        self.confusion_matrix += confusion_matrix(y_true.flatten(), y_predicted.flatten(), labels=range(self.num_classes))
        
    def calculate(self):
        metrics = {}
        # Intersection over Union (IoU) for each class
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=0) + np.sum(self.confusion_matrix, axis=1) - intersection
        iou = intersection / union.astype(np.float32)
        metrics['IoU'] = iou
        metrics['mean_IoU'] = np.nanmean(iou)
        
        # Overall Accuracy
        overall_accuracy = np.sum(intersection) / np.sum(self.confusion_matrix).astype(np.float32)
        metrics['overall_accuracy'] = overall_accuracy
        
        # Average Accuracy
        per_class_accuracy = intersection / np.sum(self.confusion_matrix, axis=1).astype(np.float32)
        metrics['average_accuracy'] = np.nanmean(per_class_accuracy)
        
        # Kappa Metric
        total = np.sum(self.confusion_matrix)
        pe = np.sum(np.sum(self.confusion_matrix, axis=0) * np.sum(self.confusion_matrix, axis=1)) / (total**2)
        po = np.sum(intersection) / total
        kappa = (po - pe) / (1 - pe)
        metrics['kappa'] = kappa
        
        return metrics