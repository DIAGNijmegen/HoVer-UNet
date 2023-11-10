import argparse
import json
import os
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from train.apply_postprocessing import apply_postprocessing
from data.pannuke_distillation_dataset import DatasetPannuke
from losses.losses import loss_fcn
from train.trainer import train

if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(prog="Train network for nuclei segmentation")
    parser.add_argument_group("Info")
    parser.add_argument('--base_project_dir', type=str, required=True)
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--experiment_group', default=0, type=int, required=True)
    parser.add_argument('--experiment_id', default=0, type=int, required=True)
    parser.add_argument('--path_train', type=str, required=True)
    parser.add_argument('--path_val', type=str, required=True)
    parser.add_argument('--path_test', type=str, required=True)
    parser.add_argument('--batch_size', default=64, type=int, choices={4, 8, 16, 32, 64, 128, 256})
    parser.add_argument('--nr_epochs', default=240, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--pannuke_path', type=str, required=True)

    parser.add_argument('--encoder', default='mit_b2',
                        help='see https://smp.readthedocs.io/en/latest/encoders.html')

    parser.add_argument('--use_true_labels', default=1, type=int, choices={0, 1})
    parser.add_argument('--use_hovernet_predictions', default=1, type=int, choices={0, 1})

    parser.add_argument('--loss_t', default=3, type=int, choices={1, 3, 5, 10, 15, 30})
    parser.add_argument('--loss_alpha', default=0.5, type=float)

    args = parser.parse_args()

    base_project_dir = args.base_project_dir
    project_name = args.project_name
    experiment_group = args.experiment_group
    experiment_id = args.experiment_id
    path_weights = None

    input_shape = (256, 256)
    output_shape = (256, 256)
    nr_types = 10
    path_train = args.path_train
    path_val = args.path_val
    path_test = args.path_test
    foldid = path_test.split('.')[0][-1]
    pannuke_path = args.pannuke_path
    true_path = f'{pannuke_path}/Fold{foldid}/masks/fold{foldid}'

    batch_size = args.batch_size
    nr_epochs = args.nr_epochs
    encoder_name = args.encoder
    lr = args.lr
    loss_alpha = args.loss_alpha
    loss_t = args.loss_t

    early_stopping = True

    use_true_labels = True
    if args.use_true_labels == 0:
        use_true_labels = False

    use_hovernet_predictions = True
    if args.use_hovernet_predictions == 0:
        use_hovernet_predictions = False

    if use_hovernet_predictions and not use_true_labels:
        loss_alpha = 1
        print("Warning: student_loss_type will be ignored")
        print("Warning: loss_alpha will be ignored")

    if use_true_labels and not use_hovernet_predictions:
        loss_alpha = 0
        loss_t = 0
        print("Warning: distill_loss_type will be ignored")
        print("Warning: loss_alpha will be ignored")
        print("Warning: loss_t will be ignored")

    if use_true_labels and use_hovernet_predictions:
        if loss_alpha == 0:
            use_hovernet_predictions = False

        if loss_alpha == 1:
            use_true_labels = False

    # Generate all path
    project_dir = os.path.join(base_project_dir, project_name)
    experiment_name = f'experiment_{experiment_group}_{experiment_id}'
    experiment_dir = os.path.join(project_dir, experiment_name)

    centroids_path = os.path.join(experiment_dir, 'centroids')
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    path_example = os.path.join(experiment_dir, 'examples')
    instance_map_path = os.path.join(experiment_dir, 'instance_maps')

    ######### DIRECTORIES SETTING ##################
    if not os.path.exists(project_dir):
        os.mkdir(project_dir)

    train_state = 0  # 0 train, 1 infer, 2 stats
    best_epoch = None
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
        os.mkdir(path_example)
        os.makedirs(centroids_path)
        os.makedirs(instance_map_path)
    else:
        exists_checkpoint = os.path.exists(checkpoint_dir)
        exists_statistics = os.path.exists(os.path.join(experiment_dir, 'statistics.json'))
        exists_tissue = os.path.exists(os.path.join(experiment_dir, 'tissue_stats.csv'))
        exists_pred_map = os.path.exists(os.path.join(experiment_dir, 'pred_masks.npy'))
        if exists_checkpoint and not exists_statistics:
            checkpoints = sorted([int(x.split('_')[-1].split('.')[0]) for x in os.listdir(checkpoint_dir)])
            checkpoint_id = checkpoints[-1]
            path_weights = os.path.join(checkpoint_dir, f'checkpoint_epoch_{checkpoint_id}.pth')
        else:
            if not exists_statistics:
                pass
            elif exists_statistics and not exists_pred_map:
                train_state = 1
                checkpoints = sorted([int(x.split('_')[-1].split('.')[0]) for x in os.listdir(checkpoint_dir)])
                best_epoch = checkpoints[-1]
            elif exists_pred_map and not exists_tissue:
                train_state = 2
            else:
                print("No operation")
                exit(-1)
    print("train state %s" % train_state)
    print("""
        Base directory: %s
        Project directory: %s
        Experiment directory: %s
        Checkpoint directory: %s
        Examples directory: %s
        Centroids directory: %s
        Instance map directory: %s
            """ % (
        base_project_dir, project_dir, experiment_dir, checkpoint_dir, path_example, centroids_path, instance_map_path))
    ###########################################################################
    with open(os.path.join(experiment_dir, 'metadata.json'), 'w') as mf:
        mf.write(json.dumps(args.__dict__))
    ##################### DATASET SETTING #####################################

    print("""
        Input shape: %s
        Output shape: %s
        Types numbers: %s
        Path train: %s
        Path validation: %s
        Path test: %s
        Pannuke path: %s
        Batch size: %s
        Epochs: %s
        Encoder: %s
                """ % (
        input_shape, output_shape, nr_types, path_train, path_val, path_test, pannuke_path, batch_size, nr_epochs,
        encoder_name))

    train_set = DatasetPannuke(path_train, mode='train', hovernet_predictions=use_hovernet_predictions,
                               true_labels=use_true_labels)
    val_set = DatasetPannuke(path_val, mode='train', hovernet_predictions=use_hovernet_predictions,
                             true_labels=use_true_labels)

    num_workers = cpu_count() // 2
    if num_workers > 8:
        num_workers = 8
    num_workers =0
    print(f'Num workers per dataloader: {num_workers}')
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers,)# prefetch_factor=2)
    dataloader_val = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                                num_workers=num_workers)#, prefetch_factor=2)
    ###################################################################################

    ############################## NETWORK AND TRAINING SETTING #####################################
    model = smp.Unet(classes=nr_types, encoder_name=encoder_name, ).to('cuda', memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    start_epoch = 0
    if path_weights:
        checkpoint_dict = torch.load(path_weights)
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        start_epoch = checkpoint_dict['epoch']
        if 'train_mode' in checkpoint_dict:
            train_mode = checkpoint_dict['train_mode']
        print(
            f"Resume model info: epoch: {checkpoint_dict['epoch']}, train_loss: {checkpoint_dict['train_loss']}, val_loss: {checkpoint_dict['val_loss']}")

    grad_scaler = torch.cuda.amp.GradScaler()

    metrics = dict()
    #########################################################################################

    with open(os.path.join(experiment_dir, 'metadata.json'), 'w') as mf:
        mf.write(json.dumps(args.__dict__))

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', min_lr=1e-7, patience=5,
                                                              threshold=1e-3)

    # LAUNCH TRAINING
    if train_state == 0:
        best_epoch = train(model, dataloader_train, dataloader_val, nr_epochs, optimizer, grad_scaler,
                           scheduler=lr_scheduler,
                           criterion=lambda x, y, z: loss_fcn(x, y, z, alpha=loss_alpha, T=loss_t),
                           metrics=metrics,
                           checkpoint_dir=Path(checkpoint_dir), experiment_dir=experiment_dir,
                           path_example=path_example,
                           early_stopping=early_stopping, start_epoch=start_epoch)
        train_state = 1

    if train_state == 1:
        path_weights = os.path.join(checkpoint_dir, 'checkpoint_epoch_{}.pth'.format(best_epoch))
        print('Path weights loaded: %s' % path_weights)
        res = apply_postprocessing(path_weights=path_weights, path_test=path_test, model=model)
        _, post_processed, file_names, masks = zip(*res)
        print('Saving pred_masks.npy')
        np.save(os.path.join(experiment_dir, 'pred_masks.npy'), np.array(masks))
        print('pred_masks.npy saved')
        print("Saving centroids and pred maps")
        for pp, file_name in zip(post_processed, file_names):
            np.save(os.path.join(instance_map_path, file_name), pp[0])
            with open(os.path.join(centroids_path, file_name.replace('npy', 'json')), 'w') as fc:
                centroids = {}
                for k, v in pp[1].items():
                    centroids[int(k)] = {
                        'bbox': v['bbox'].tolist(),
                        'centroids': v['centroid'].tolist(),
                        'contours': v['contour'].tolist(),
                        'type_prob': float(v['type_prob']),
                        'type': int(v['type'])
                    }
                fc.write(json.dumps(centroids))
                fc.close()
        print("centroids and pred maps  save")
        train_state = 2

    if train_state == 2:
        print(
            f'RUN: python3 pannuke_metrics/run.py --true_path {true_path} --pred_path {experiment_dir} --save_path {experiment_dir}')
        os.system(
            f'python3 pannuke_metrics/run.py --true_path {true_path} --pred_path {experiment_dir} --save_path {experiment_dir}')
        print(
            f'END: python3 pannuke_metrics/run.py --true_path {true_path} --pred_path {experiment_dir} --save_path {experiment_dir}')
