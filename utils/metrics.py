import numpy as np
# import torch

class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        # self.confusion_matrix_torch = torch.zeros((num_classes, num_classes))

    def process_batch(self, preds, targets):
        preds = preds.argmax(1).cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        mask = (targets >= 0) & (targets < self.num_classes)
        
        confusion_mtx = np.bincount(self.num_classes * targets[mask].astype(int) + preds[mask],
                        minlength=self.num_classes ** 2)
        confusion_mtx = confusion_mtx.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += confusion_mtx

    def print(self):
        for i in range(self.num_classes):
            print(f"Class {i}: {self.confusion_matrix[i, i]} / {self.confusion_matrix[i].sum()}")

    def get_pix_acc(self):
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

    def get_class_acc(self):
        class_acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return np.nanmean(class_acc)
    
    def get_iou(self):
        divisor = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - \
                    np.diag(self.confusion_matrix)
        iou = np.diag(self.confusion_matrix) / divisor
        return iou
    
    def get_mean_iou(self):
        iou = self.get_iou()
        return np.nansum(iou) / self.num_classes
    
    # def sync(self, device):
    #     self.confusion_matrix_torch = torch.from_numpy(self.confusion_matrix).to(device)
    #     dist.all_reduce(self.confusion_matrix_torch, op=dist.ReduceOp.SUM)
    #     self.confusion_matrix = self.confusion_matrix_torch.cpu().numpy()