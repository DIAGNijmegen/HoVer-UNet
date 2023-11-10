"""
This script process pannuke dataset to organize the data for distillation. It is required to train FastHoVerNet
1. Add hovernet gt
2. Add hovernet predictions
3. Split data in one file for each patch (to have one file require too much ram)
4. Create one h5 file for each fold

NOTE: This script can be improved because it makes inference for each image without batching.
One pannuke image is patched because the output shape are not equal to input shape

I didn't improve it because I run this script one time only to create dataset.
This script run in about 1 hour.
"""
import argparse
import shutil

import torch
from patchify import patchify

from data.utils import remap_labels, unpatchify
from models.HoVerNet.net_desc import create_model
from models.HoVerNet.targets import gen_targets

import os
from glob import glob

import h5py
import numpy as np
from tqdm import tqdm


def multiple_to_one(src_path, dst_path):
    dt = h5py.special_dtype(vlen=str)
    dim = 200
    list_files = glob(os.path.join(src_path, '*.npy'))
    list_files = [(int(x.split('/')[-1].split('_')[0]), x) for x in list_files]

    list_files.sort(key=lambda x: x[0])
    _, list_files = zip(*list_files)
    list_objects = []
    types = []
    if os.path.exists(dst_path):
        os.remove(dst_path)
    h5file = h5py.File(dst_path, 'w')

    h5_images = h5file.create_dataset("images", shape=(0, 256, 256, 3), maxshape=(None, 256, 256, 3))
    h5_hovernet = h5file.create_dataset("hovernet_predictions", shape=(0, 256, 256, 10), maxshape=(None, 256, 256, 10))
    h5_true = h5file.create_dataset("true_labels", shape=(0, 256, 256, 9), maxshape=(None, 256, 256, 9))
    h5_types = h5file.create_dataset("types", shape=(0,), maxshape=(None,), dtype=dt)
    progress = tqdm(total=len(list_files))
    for file in list_files:
        if len(list_objects) % dim == 0 and len(list_objects) > 0:
            h5_images.resize(h5_images.shape[0] + dim, axis=0)
            h5_hovernet.resize(h5_hovernet.shape[0] + dim, axis=0)
            h5_true.resize(h5_true.shape[0] + dim, axis=0)
            h5_types.resize(h5_types.shape[0] + dim, axis=0)

            data = np.array(list_objects)

            images, masks = data[..., :3], data[..., 3:]
            gt, hovernet = masks[..., :9], masks[..., 9:]

            h5_images[-dim:] = images
            h5_hovernet[-dim:] = hovernet
            h5_true[-dim:] = gt
            h5_types[-dim:] = types

            list_objects = []
            types = []
        list_objects.append(np.load(file))
        types.append(file.split('/')[-1].split('_')[-1].replace('.npy', ''))
        progress.update(1)
    progress.close()

    if len(list_objects) > 0:
        h5_images.resize(h5_images.shape[0] + len(list_objects), axis=0)
        h5_hovernet.resize(h5_hovernet.shape[0] + len(list_objects), axis=0)
        h5_true.resize(h5_true.shape[0] + len(list_objects), axis=0)
        h5_types.resize(h5_types.shape[0] + len(list_objects), axis=0)

        data = np.array(list_objects)

        images, masks = data[..., :3], data[..., 3:]
        gt, hovernet = masks[..., :9], masks[..., 9:]

        h5_images[-len(list_objects):] = images
        h5_hovernet[-len(list_objects):] = hovernet
        h5_true[-len(list_objects):] = gt
        h5_types[-len(list_objects):] = types

    h5file.close()


def main(parser):
    args = parser.parse_args()
    pannuke_weights_path = args.pannuke_weights_path
    pannuke_path = args.pannuke_path
    save_path = args.save_path

    hovernet = create_model(mode='fast', nr_types=6)
    hovernet.to('cuda')
    hovernet.load_state_dict(torch.load(pannuke_weights_path)['desc'])
    hovernet.eval()

    for fold in [1, 2, 3]:
        print("Running fold ", fold)

        path_types = os.path.join(pannuke_path, 'Fold%s/images/fold%s/types.npy' % (fold, fold))
        path_images = os.path.join(pannuke_path, 'Fold%s/images/fold%s/images.npy' % (fold, fold))
        path_masks = os.path.join(pannuke_path, 'Fold%s/masks/fold%s/masks.npy' % (fold, fold))

        path_save_tmp = os.path.join(save_path, 'tmp')
        if not os.path.exists(path_save_tmp):
            os.makedirs(path_save_tmp)
        types = np.load(path_types)
        images = np.load(path_images)
        masks = np.load(path_masks)

        progess = tqdm(total=images.shape[0])
        for i in range(images.shape[0]):
            img_type = types[i]
            mask = masks[i]
            img = images[i]
            ################## HOVERNET PREDICTIONS ########################################
            img_pred = np.copy(img)
            mask_pred = np.copy(mask)
            mask = np.concatenate((mask_pred[..., -1, None], mask_pred[..., :-1]), axis=-1)
            img_pred = np.lib.pad(img_pred, ((0, 72), (0, 72), (0, 0)), mode='reflect')
            patches = patchify(img_pred, (164, 164, 3), step=82)
            x = np.reshape(patches, (patches.shape[0] * patches.shape[1],) + patches.shape[3:])
            x = np.pad(x, ((0, 0), (46, 46), (46, 46), (0, 0)), mode='reflect')
            x = torch.from_numpy(x.astype('float32')).to(
                'cuda')
            x = torch.permute(x, (0, 3, 1, 2))
            with torch.no_grad():
                res = hovernet(x)
            hv_map = torch.permute(res['hv'], (0, 2, 3, 1)).cpu()
            tp_map = torch.permute(res['tp'], (0, 2, 3, 1)).cpu()
            np_map = torch.permute(res['np'], (0, 2, 3, 1)).cpu()
            final = np.concatenate((np_map, hv_map, tp_map), axis=-1)
            re = unpatchify(np.squeeze(final), (164, 164), 82, (328, 328))
            re = re[:328 - 72, :328 - 72, :]
            ##################################################################################

            ######## GT HOVERNET ##########################################################
            instance_mask = remap_labels(np.copy(mask))
            ann = gen_targets(instance_mask, img.shape[:-1])
            mask[mask > 1] = 1
            gt = np.concatenate((ann['np_map'][..., None], ann['hv_map'], mask), axis=-1)

            #################################################################################
            np.save(os.path.join(path_save_tmp, '%s_%s.npy' % (i, img_type.lower())),
                    np.concatenate((img, gt, re), axis=-1))
            progess.update(1)

        progess.close()
        h5_file_path = os.path.join(save_path, f'fold{fold}.h5')
        multiple_to_one(path_save_tmp, h5_file_path)
        shutil.rmtree(path_save_tmp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Create pannuke dataset for distillation")

    parser.add_argument("--pannuke_path", type=str, required=True)

    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--pannuke_weights_path", type=str, required=True)

    main(parser)
