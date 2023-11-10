import gc
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from data.augmentation import get_augmentation_gpu
from misc.early_stopping import EarlyStopping

torch.manual_seed(0)


def get_lr(apt):
    for param_group in apt.param_groups:
        return param_group['lr']


def compute_metrics_train(pred, true, nr_types, metrics, epoch_metrics: dict, mode='train'):
    metrics_results = dict()
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    true -= 1
    pred -= 1
    tp, fp, fn, tn = smp.metrics.get_stats(
        output=pred,
        target=true.type(torch.long),
        mode='multiclass',
        num_classes=nr_types - 1,
        ignore_index=-1,
    )

    for metric in metrics:
        value = metric(tp, fp, fn, tn, reduction='micro').item()
        metrics_results[str(metric.__name__) + '_' + mode] = value
        epoch_metrics[str(metric.__name__) + '_' + mode].append(value)

    return metrics_results, epoch_metrics


def train_step(model: torch.nn.Module,
               train_set: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
               grad_scaler: torch.cuda.amp.GradScaler,
               criterion: torch, metrics: dict, progress: tqdm.tqdm, nr_types, gpu_augmentation=None):
    model.train()
    losses = []
    metrics_epoch = defaultdict(list)
    for i, (x, y) in enumerate(train_set):
        x = x.to('cuda', non_blocking=True, memory_format=torch.channels_last)
        y = y.to('cuda', non_blocking=True, memory_format=torch.channels_last)
        x = torch.permute(x, (0, 3, 1, 2)).contiguous()
        y = torch.permute(y, (0, 3, 1, 2)).contiguous()
        if gpu_augmentation is not None:
            x, y = gpu_augmentation(x, y)

        # todo too static
        if y.shape[1] == 19:
            z, y = y[:, :9, ...], y[:, 9:, ...]
        elif y.shape[1] == 10:
            z = None
        elif y.shape[1] == 9:
            z = y
            y = None
        else:
            raise Exception()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(x)
            loss = criterion(outputs, y, z)

        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        if len(metrics) > 0:
            metrics_step, metrics_epoch = compute_metrics_train(pred=outputs,
                                                                true=y,
                                                                nr_types=nr_types,
                                                                metrics=metrics,
                                                                epoch_metrics=metrics_epoch,
                                                                mode='train')
        else:
            metrics_step = {}
            metrics_epoch = {}
        progress.set_postfix(loss='{:05.3f}'.format(loss.item()), **metrics_step, lr=get_lr(optimizer))
        losses.append(loss.item())
        progress.update(1)

        torch.cuda.empty_cache()
        _ = gc.collect()

    return statistics.mean(losses), {k: statistics.mean(v) for k, v in metrics_epoch.items()}


def val_step(model: torch.nn.Module, val_set: torch.utils.data.DataLoader, criterion, metrics, nr_types):
    model.eval()
    losses = []
    metrics_epoch = defaultdict(list)
    example = None
    for i, (x, y) in enumerate(val_set):
        x = x.to('cuda', non_blocking=True, memory_format=torch.channels_last)
        y = y.to('cuda', non_blocking=True, memory_format=torch.channels_last)
        x = torch.permute(x, (0, 3, 1, 2)).contiguous()
        y = torch.permute(y, (0, 3, 1, 2)).contiguous()
        # todo too static
        if y.shape[1] == 19:
            z, y = y[:, :9, ...], y[:, 9:, ...]
        elif y.shape[1] == 10:
            z = None
        elif y.shape[1] == 9:
            z = y
            y = None
        else:
            raise Exception()

        with torch.no_grad():
            outputs = model(x)
            loss = criterion(outputs, y, z)
        losses.append(loss.item())
        if len(metrics) > 0:
            _, metrics_epoch = compute_metrics_train(pred=outputs,
                                                     true=y,
                                                     nr_types=nr_types,
                                                     metrics=metrics,
                                                     epoch_metrics=metrics_epoch,
                                                     mode='val')
        else:
            metrics_epoch = {}
        if i == len(val_set) - 1:
            if y is not None:
                example = zip(x, y, outputs)
            if z is not None:
                example = zip(x, z, outputs)

        torch.cuda.empty_cache()
        _ = gc.collect()
    return statistics.mean(losses), {k: statistics.mean(v) for k, v in metrics_epoch.items()}, example


def save_example(ex, path):
    fig, ax = plt.subplots(5, 9, figsize=(25, 15))
    for ix, (im, true, pred) in enumerate(ex):
        true_np = F.softmax(true[:2], dim=0)[1]
        true_h, true_v = true[2], true[3]
        true_tp = torch.argmax(F.softmax(true[4:], dim=0), dim=0)

        pred_np = F.softmax(pred[:2], dim=0)[1]
        pred_h, pred_v = pred[2], pred[3]
        pred_tp = torch.argmax(F.softmax(pred[4:], dim=0), dim=0)

        if ix == 5:
            break
        ax[ix][0].imshow(torch.permute(im.cpu(), (1, 2, 0)).type(torch.float))
        ax[ix][1].imshow(true_np.cpu())
        ax[ix][2].imshow(true_h.cpu())
        ax[ix][3].imshow(true_v.cpu())
        ax[ix][4].imshow(true_tp.cpu(), vmin=0, vmax=5)
        ax[ix][5].imshow(pred_np.cpu())
        ax[ix][6].imshow(pred_h.cpu())
        ax[ix][7].imshow(pred_v.cpu())
        ax[ix][8].imshow(pred_tp.cpu(), vmin=0, vmax=5)

        ax[ix][0].axis('off')
        ax[ix][1].axis('off')
        ax[ix][2].axis('off')
        ax[ix][3].axis('off')
        ax[ix][4].axis('off')
        ax[ix][5].axis('off')
        ax[ix][6].axis('off')
        ax[ix][7].axis('off')
        ax[ix][8].axis('off')
    fig.savefig(path, dpi=300, format='png')
    plt.close(fig)




def train(model: torch.nn.Module, train_set: torch.utils.data.DataLoader, val_set: torch.utils.data.DataLoader,
          epochs: int, optimizer: torch.optim.Optimizer, grad_scaler: torch.cuda.amp.GradScaler, scheduler,
          criterion, metrics, checkpoint_dir=None, experiment_dir=None, path_example=None,
          early_stopping=None,start_epoch=None):

    statistics_training = dict()
    best_loss = None
    if early_stopping:
        early_stopper = EarlyStopping(patience=10)
    else:
        early_stopper = None

    if start_epoch is None:
        start_epoch = 0
    best_epoch = start_epoch
    nr_types = 10
    augmentation = get_augmentation_gpu().cuda()

    for epoch in range(start_epoch, epochs):
        statistics_epoch = {}
        print("\nEpoch %s/%s - best epoch %s" % (epoch + 1, epochs, best_epoch))
        with tqdm.tqdm(total=len(train_set), unit='step',
                       ncols=100 + (50 * len(metrics)),
                       bar_format='{desc}{n_fmt}/{total_fmt}|{bar}|ETA:{remaining} '
                                  '- {elapsed} {rate_inv_fmt}{postfix}') as progress_bar:
            train_loss, train_metrics = train_step(model, train_set, optimizer, grad_scaler, criterion, metrics,
                                                   progress_bar, nr_types=nr_types, gpu_augmentation=augmentation)
            val_loss, val_metrics, example = val_step(model, val_set, criterion, metrics, nr_types=nr_types)

            statistics_epoch['val_loss'] = val_loss
            statistics_epoch['train_loss'] = train_loss

            progress_bar.set_postfix(loss='{:05.3f}'.format(train_loss), loss_val='{:05.3f}'.format(val_loss),
                                     **val_metrics, **train_metrics, lr=get_lr(optimizer))

            if scheduler:
                scheduler.step(val_loss)

            if best_loss is None:
                best_loss = val_loss
                best_epoch = epoch + 1
            else:
                if best_loss > val_loss:
                    best_loss = val_loss
                    best_epoch = epoch + 1

            if best_epoch == epoch + 1:
                if checkpoint_dir:
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    torch.save({  # Save our checkpoint loc
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': train_loss,
                    },
                        str(checkpoint_dir / 'checkpoint_epoch_{}.pth'.format(epoch + 1)))
                if path_example:
                    save_example(example, os.path.join(path_example, 'ex_epoch_{}.png'.format(epoch + 1)))

            statistics_training[epoch] = {**statistics_epoch, **train_metrics, **val_metrics}
            if early_stopping:
                early_stopper(val_loss, model)

    with open(os.path.join(experiment_dir, 'statistics.json'), 'w') as file:
        file.write(json.dumps(statistics_training))
    return best_epoch
