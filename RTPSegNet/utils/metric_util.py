import numpy as np
from pytorch_lightning.metrics import Metric
from dataloader.pc_dataset import get_label_name


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label,learning_map=None):
    if learning_map is not None:
        output = np.vectorize(learning_map.__getitem__)(output)
        target = np.vectorize(learning_map.__getitem__)(target)

    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist


class IoU(Metric):
    def __init__(self, dataset_config, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.hist_list = []
        self.best_miou = 0
        self.label_name, self.learning_map = get_label_name(dataset_config["label_mapping"])
        #if dataset_config["learning_map"]==False:
        #    self.learning_map = None
        self.unique_label = np.asarray(sorted(list(self.label_name.keys())))[1:] - 1
        self.unique_label_str = [self.label_name[x] for x in self.unique_label + 1]

    def update(self, predict_labels, val_pt_labs) -> None:
        self.hist_list.append(fast_hist_crop(predict_labels, val_pt_labs, self.unique_label,  self.learning_map))

    def compute(self):
        iou = per_class_iu(sum(self.hist_list))
        if np.nanmean(iou) > self.best_miou:
            self.best_miou = np.nanmean(iou)
        self.hist_list = []
        return iou, self.best_miou