import colorsys
import json
import os
import random
from multiprocessing import Pool
from time import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pannuke_distillation_dataset import DatasetPannuke
from models.HoVerNet.post_proc import process


def random_colors(N, bright=True):
    """Generate random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def visualize_instances_dict(
        input_image, inst_dict, draw_dot=False, type_colour=None, line_thickness=2
):
    """Overlays segmentation results (dictionary) on image as contours.

    Args:
        input_image: input image
        inst_dict: dict of output prediction, defined as in this library
        draw_dot: to draw a dot for each centroid
        type_colour: a dict of {type_id : (type_name, colour)} ,
                     `type_id` is from 0-N and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours
    """
    # overlay = np.copy((input_image))
    overlay = np.zeros(input_image.shape)
    inst_rng_colors = random_colors(len(inst_dict))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colour is not None:
            inst_colour = type_colour[inst_info["type"]][1]
        else:
            inst_colour = (inst_rng_colors[idx]).tolist()
        cv2.drawContours(overlay, [inst_contour], -1, inst_colour, line_thickness)

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay


def create_mask(x, _pred):
    instance_map, _centroids = x
    mask = np.zeros(instance_map.shape + (6,))

    for idx, info in _centroids.items():
        try:
            mask[..., info['type']][instance_map == idx] = idx
            mask[..., 0][instance_map == idx] = 1
        except Exception:
            print(_pred[-1])
    return mask


def _postprocess(_pred):
    x = process(_pred[1], nr_types=6)
    mask = create_mask(x, _pred)
    return _pred[0], x, _pred[2], mask


def apply_postprocessing(path_weights, path_test, model):
    model.load_state_dict(torch.load(path_weights)['model_state_dict'])
    model.eval()
    data_infer = DatasetPannuke(path_test, mode='infer')
    dataloader = DataLoader(data_infer, shuffle=False, pin_memory=True, num_workers=0, batch_size=64)
    predictions = []
    with tqdm(total=len(dataloader)) as progress:
        for bn, (im, info) in enumerate(dataloader):
            t0 = time()
            im = im.to('cuda')
            im = torch.permute(im, (0, 3, 1, 2)).contiguous()
            with torch.no_grad():
                pred = model(im)
                pred_np = F.softmax(pred[:, :2, ...], dim=1)[:, 1, ...].to('cpu')
                pred_h, pred_v = pred[:, 2, ...].to('cpu'), pred[:, 3, ...].to('cpu')

                pred_tp = torch.argmax(F.softmax(pred[:, 4:, ...], dim=1), dim=1).to('cpu')
            pred_map = np.concatenate((pred_tp[..., None], pred_np[..., None], pred_h[..., None], pred_v[..., None]),
                                      axis=-1)
            im = torch.permute(im, (0, 2, 3, 1)).contiguous().to('cpu').numpy()
            progress.set_postfix(time=time() - t0)
            predictions.extend(zip(list(im), list(pred_map), list(info)))
            progress.update(1)
    progress.close()

    progress = tqdm(total=len(predictions))
    results = []
    for prediction in predictions:
        results.append(_postprocess(prediction))
        progress.update(1)
    progress.close()

    return results
