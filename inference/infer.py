import os
import pickle
from glob import glob
from typing import Tuple

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from patchify import patchify, unpatchify
from torch.utils.data import TensorDataset, DataLoader

from models.HoVerNet.post_proc import process


class FastHoVerNetInfer:

    def __init__(self, images_path: str, weights_path: str, save_path: str, path_size: Tuple[int, int, int], step: int,
                 ext: str = 'png', overlay: bool = True):
        self.fasthovernet = smp.Unet(encoder_name='mit_b2', classes=10, in_channels=3).cuda()

        self.fasthovernet.load_state_dict(torch.load(weights_path)['model_state_dict'])
        self.fasthovernet.eval()

        self.save_path_predictions = os.path.join(save_path, 'predictions')
        self.save_path_overlay = os.path.join(save_path, 'overlay')
        if not os.path.exists(self.save_path_predictions):
            os.makedirs(self.save_path_predictions)
        if not os.path.exists(self.save_path_overlay):
            os.makedirs(self.save_path_overlay)

        self.patch_size = path_size
        self.step = step
        self.images_paths = glob(os.path.join(images_path, f"*.{ext}"))
        self.overlay = overlay

    def _gen_patches(self, image, idx) -> (np.array, Tuple[Tuple[int, int, int], int, int], np.array):
        h, w, _ = image.shape
        n_patches = np.ceil(
            (np.array(image.shape) - np.array(self.patch_size)) / np.array([self.step, ] * image.ndim) + 1)
        image_shape = (n_patches - 1) * (np.array([self.step, ] * image.ndim)) + np.array(self.patch_size)
        w_pad, h_pad, _ = (image_shape - np.array(image.shape)) / 2
        if w_pad != int(w_pad):
            w_pad = (int(w_pad) + 1, int(w_pad))
        else:
            w_pad = (int(w_pad), int(w_pad))
        if h_pad != int(h_pad):
            h_pad = (int(h_pad) + 1, int(h_pad))
        else:
            h_pad = (int(h_pad), int(h_pad))
        pad_image = np.pad(image, (w_pad, h_pad, (0, 0)), mode='reflect')
        patches = patchify(pad_image, self.patch_size, step=self.step)
        return patches, (pad_image.shape, w_pad, h_pad, patches.shape[:2], idx)

    @staticmethod
    def _marge_patches(patches, info):
        pad_image_shape, w_pad, h_pad, grid = info
        pad_image_shape = np.asarray(pad_image_shape)
        pad_image_shape[-1] = patches.shape[-1]
        pad_image_shape = tuple(pad_image_shape)
        patches = patches.reshape((grid + (1,) + patches.shape[-3:])).cpu().numpy()
        img = unpatchify(patches, pad_image_shape)
        if h_pad[1] == 0:
            h_pad = slice(h_pad[0], img.shape[1])
        else:
            h_pad = slice(h_pad[0], -h_pad[1])

        if w_pad[1] == 0:
            w_pad = slice(w_pad[0], img.shape[0])
        else:
            w_pad = slice(w_pad[0], -w_pad[1])
        img = img[w_pad, h_pad, :]

        return img

    def _post_process(self, pred):
        pred_inst, nuclei_dict = process(pred, nr_types=6, return_centroids=True)
        nuc_val_list = list(nuclei_dict.values())
        nuc_uid_list = np.array(list(nuclei_dict.keys()))
        nuc_type_list = np.array([v["type"] for v in nuc_val_list])
        nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])
        nuc_bbox_list = np.array([v['bbox'] for v in nuc_val_list])
        nuc_types_prob = np.array([v['type_prob'] for v in nuc_val_list])
        nuc_cont_list = [v['contour'] for v in nuc_val_list]

        mat = {
            'inst_map': pred_inst,
            'inst_uid': nuc_uid_list,
            'inst_type': nuc_type_list,
            'inst_centroid': nuc_coms_list,
            'inst_bbox': nuc_bbox_list,
            'inst_type_prod': nuc_types_prob,
            'inst_contour': nuc_cont_list,
        }
        return mat

    def _overlay_prediction(self, mat, file_name, image):
        colors = {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
            4: (255, 255, 0),
            5: (255, 0, 255),
        }

        inst_contour = mat['inst_contour']
        inst_centroid = mat['inst_centroid'].astype('int32')
        inst_type = mat['inst_type']

        for i, contour in enumerate(inst_contour):
            cv2.drawContours(image, [contour], -1, colors[inst_type[i]], 2)
            cv2.circle(image, inst_centroid[i], 3, colors[inst_type[i]], -1)

        Image.fromarray(image.astype('uint8')).save(
            os.path.join(self.save_path_overlay, f"{file_name.split('.')[0]}.jpg"))

    def infer(self):
        # linferred = []
        patches_list = []
        info_list = []
        images = []
        idx = 0
        while len(self.images_paths) > 0:
            path_image = self.images_paths.pop()
            image_name = os.path.basename(path_image).split('.')[0]
            print(f"{image_name}")
            image = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_BGR2RGB)
            patches, info = self._gen_patches(image, idx)
            idx += 1
            num_patches = patches.shape[0] * patches.shape[1]
            patches = patches.reshape((num_patches,) + patches.shape[3:])
            info_list.append(info + (image_name, num_patches,))
            patches_list.extend(patches.tolist())
            images.append(image)
            if len(patches_list) > 500 or len(self.images_paths) == 0:
                tensor_x = torch.Tensor(patches_list).cuda()  # transform to torch tensor
                tensor_x = tensor_x.permute(0, 3, 1, 2) / 255
                mem_dataset = TensorDataset(tensor_x)  # create your dataset
                loader = DataLoader(mem_dataset, shuffle=False, batch_size=64)
                preds = []
                for x in loader:
                    x = x[0]
                    with torch.no_grad():
                        pred = self.fasthovernet(x)
                        preds.append(pred)
                preds = torch.cat(preds, dim=0).permute(0, 2, 3, 1).contiguous()
                pred_np = F.softmax(preds[..., :2], dim=-1)[..., 1][..., None]
                pred_hv = preds[..., 2:4]
                pred_tp = torch.argmax(F.softmax(preds[..., 4:], dim=-1), dim=-1)[..., None]
                preds = torch.cat([pred_tp, pred_np, pred_hv], dim=-1)
                r = 0
                for i, info in enumerate(info_list):
                    num_p = info[-1]
                    pred_i = preds[r:r + num_p]
                    r += num_p
                    pred_i = self._marge_patches(pred_i, info[:4])

                    mat = self._post_process(pred_i)
                    with open(os.path.join(self.save_path_predictions, f"{info[-2]}.pickle"), 'wb') as f:
                        pickle.dump(mat, f, protocol=pickle.HIGHEST_PROTOCOL)
                    # inferred.append(mat)
                    if self.overlay:
                        self._overlay_prediction(mat, info[-2], images[i])
                patches_list = []
                info_list = []
                images = []
        # return inferred
