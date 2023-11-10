import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt

from data.augmentation import get_augmentation_gpu
import torch.nn.functional as F

class DatasetPannuke(Dataset):
    """
    Distillaton pannuke dataset
    """

    def __init__(self, path: str, mode: str = 'train', true_labels: bool = False,
                 hovernet_predictions: bool = True):
        """
        :param path: path of processed pannuke dataset, h5 file
        :param mode: train or infer
        :param true_labels: load ground truth
        :param hovernet_predictions: load hovernet predictions
        """
        assert isinstance(path, str), "path have be instance of string"
        assert isinstance(mode, str) and mode in ['train', 'infer'], "mode must be either train or infer"
        assert isinstance(hovernet_predictions, bool) and isinstance(true_labels, bool) and (
                hovernet_predictions or true_labels), \
            "hovernet_predictions and true_labels must be boolean, and at least one must be true"

        self.path = path
        self.input_shape = (256, 256)
        self.output_shape = (256, 256)
        self.nr_types = 6
        self.mode = mode
        self.true_labels = true_labels
        self.hovernet_predictions = hovernet_predictions
        data = h5py.File(path, 'r')
        self.images = data['images']
        if mode == 'infer':
            self.types = data['types']
        if true_labels:
            self.labels = data['true_labels']
        if hovernet_predictions:
            self.hovernet = data['hovernet_predictions']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        outputs = ((self.images[idx] / 255).astype('float32'),)
        if self.mode == 'train':
            if self.true_labels:
                outputs += (self.labels[idx].astype('float32'),)
            if self.hovernet_predictions:
                outputs += (self.hovernet[idx].astype('float32'),)
            if len(outputs) == 3:
                outputs = (outputs[0], np.concatenate(outputs[1:], axis=-1))
        elif self.mode == 'infer':
            outputs += ('%s_%s' % (idx, self.types[idx].decode('utf8')),)

        return outputs


if __name__ == '__main__':

    path_train = '/work/cristian/test_fasthovernet/data/fold2.h5'

    dataset = DatasetPannuke(path_train, mode='train', true_labels=True, hovernet_predictions=True)

    dataloader = DataLoader(dataset, num_workers=0, shuffle=False, batch_size=16)

    aug = get_augmentation_gpu().cuda()
    fig, ax = plt.subplots(2, 9, figsize=(20, 5))


    for idx, (x, y) in enumerate(dataloader):
        x = x.cuda()
        y = y.cuda()
        x = torch.permute(x, (0, 3, 1, 2)).contiguous()
        y = torch.permute(y, (0, 3, 1, 2)).contiguous()
        x, y = aug(x, y)

        y, z = y[:, :9, ...], y[:, 9:, ...]
        x = torch.permute(x, (0, 2, 3, 1))
        y = torch.permute(y, (0, 2, 3, 1))
        z = torch.permute(z, (0, 2, 3, 1))

        im = x[0].cpu()
        ax[idx][0].imshow(im)

        gt = y[0].cpu()
        gt_np, gt_hv, gt_tp = gt[..., 0], gt[..., 1:3], gt[..., 3:]
        gt_tp = torch.argmax(gt_tp, dim=-1)
        hov_pred = z[0].cpu()
        hov_np, hov_hv, hov_tp = hov_pred[..., :2], hov_pred[..., 2:4], hov_pred[..., 4:]
        hov_np = F.softmax(hov_np, dim=-1)[..., 1]
        hov_tp = torch.argmax(F.softmax(hov_tp, dim=-1), dim=-1)
        ax[idx][1].imshow(gt_np)
        ax[idx][2].imshow(gt_tp, vmin=0, vmax=5)
        ax[idx][3].imshow(gt_hv[..., 0])
        ax[idx][4].imshow(gt_hv[..., 1])
        ax[idx][5].imshow(hov_np)
        ax[idx][6].imshow(hov_tp, vmin=0, vmax=5)
        ax[idx][7].imshow(hov_hv[..., 0])
        ax[idx][8].imshow(hov_hv[..., 1])

        for i in range(9):
            ax[idx][i].axis('off')

        if idx == 1:
            break
    fig.show()
