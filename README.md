# "HoVer-UNet": Accelerating HoVerNet with U-Net-based multi-class nuclei segmentation via knowledge distillation
This repository contains the original **PyTorch** implementation of **HoVer-UNet,** an approach to distill the knowledge of the multi-branch HoVerNet framework for nuclei instance segmentation and classification in histopathology.

![](docs/network.png)

## Features

* [x] Training
* [x] Tile Inference
* [ ] WSI Inference

## Dataset
PanNuke [Fold1](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip), [Fold2](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip), [Fold3](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip) <br />
CoNSeP [Download](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/)

## Set Up Environment

To run this code set up python environment as follows:

```
https://github.com/DIAGNijmegen/HoVer-UNet.git
cd HoVer-UNet
python -m venv hoverunet_env
source hoverunet_env/bin/activate
pip install -r requirements.txt
```

We tested our code with `Python 3.8` and `Cuda 12`.

## Repository Structure

Below are the main directories in the repository: 

- `data/`: the data loader and augmentation
- `pannuke_metrics/`: scripts for metric calculation provided by [Link](https://github.com/TissueImageAnalytics/PanNuke-metrics)
- `misc/`: utils that are
- `models/`: model definition
- `losses/`: losses functions used during the traning
- `train/`:  all training functions
- `inference/`:  all inference functions and classes
- `docs/`: figures/GIFs used in the repo


Below are the main executable scripts in the repository:

- `process_pannuke_dataset.py`: script to transform PanNuke dataset for knowledge distillation
- `run_train.py`: main training script
- `run_infer.py`: main inference script for tile 
- `run_train.sh`: bash script to run training


## Running the Code
### Training

#### Data Format
To train HoVer-UNet on PanNuke dataset, execute `process_pannuke_dataset.py` to transform in hdfs file where for each image it contains:

- `Original Image in RGB`
- `Ground truth`: nuclei map (one channel), horizontal and vertical map (two channels), and type map (N channels, where N is number of classes including background)
- `HoVerNet predictions`: nuclei map (two channel), horizontal and vertical map (two channels), and type map (N channels, where N is number of classes including background)

```
python process_pannuke_dataset.py --pannuke_path PANNUKE_PATH 
                                  --save_path SAVE_PATH
                                  --pannuke_weights_path WEIGHTS_PANNUKE_PATH
```
Options:
```
--pannuke_path              PanNuke dataset path in the same downloaded structure as provided by ""
--save_path                 path where you want save processed dataset
--pannuke_weights_path      PanNuke weights 
```
`--pannuke_path` must have the tree folder:
```
pannuke
    -- Fold1
    -- Fold2
    -- Fold3
```
And each fold must be the same to downloaded one.

Download `pannuke_weights` [here](https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view?pli=1). <br />
You can also download pre-processed PanNuke dataset [here](https://zenodo.org/records/10103514?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjNmOWIwNmM3LWYwZmUtNGRhZi05MDY3LWQ2MjY1MzcwODQwYyIsImRhdGEiOnt9LCJyYW5kb20iOiJhNzNiODFlY2QxODBiNzRhYmY2ZjVmMGI0NTc4NDJiNyJ9.ajdCHU_PoLsEr0L-KWqcs-dAI_gNDmtStCylS6sFa8nos8bvaV7dzgTs30d-CrkuA_Aqm8431va5OYGwNx_EbQ)

#### Run Training
To train HoVer-UNet run `run_train.py` as fallows:
```
python run_train.py --base_project_dir BASE_PROJECT_DIR 
                --project_name PROJECT_NAME 
                --experiment_group EXPERIMENT_GROUP 
                --experiment_id EXPERIMENT_ID 
                --path_train PATH_TRAIN 
                --path_val PATH_VAL
                --path_test PATH_TEST
                --pannuke_path PANNUKE_PATH
                [--batch_size {32,64,128,256,4,8,16}]
                [--nr_epochs NR_EPOCHS] [--lr LR]
                [--encoder ENCODER]
                [--use_true_labels {0,1}]
                [--use_hovernet_predictions {0,1}]
                [--loss_t {1,3,5,10,15,30}]
                [--loss_alpha LOSS_ALPHA]
```

Options:

```
--base_project_dir                              base path where save the exprtiment
--project_name                                  name of project
--experiment_group                              eperiment groupd id, group results like cross validation
--experiment_id                                 experiment id
--path_train                                    path of PanNuke processed h5 file for training
--path_val                                      path of PanNuke processed h5 file for validation
--path_test                                     path of PanNuke processed h5 file for test
--pannuke_path                                  path of orginal PanNuke dataset (metrics calculation)
[--batch_size {32,64,128,256,4,8,16}]           batch size
[--nr_epochs]                                   numebr of epochs
[--lr LR]                                       initial learning rate
[--encoder]                                     encoder name to use as U-Net backbone 
[--use_true_labels {0,1}]                       use ground thruth or no (1 is use distillation)
[--use_hovernet_predictions {0,1}]              use hovernet predictions or no (1 is use alpha not equal 0)
[--loss_t {1,3,5,10,15,30}]                     temperature coefficient to compute KL-Divergence between student predictions and hovernet predictions
[--loss_alpha]                                  alpha values to combina student losss and distillation loss
```

`--base_project_dir, --project_name, --experiment_group, and --experiment_id` options are used to manage different grouped experiments due to the configuration suggested by PanNuke authors. 


You can also use bash script and manually modify the options in `run_train.sh`
```
#! /bin/bash
./run_train.sh
```

`run_train.py` makes training on PanNuke (using training set and validation set) and then tests the trained model on test set.
It saves the best checkpoints during the training, and for each one saves some examples as well. For testing, it save predictions and also compute the metrics. 
`run_train.sh` makes training on PanNuke three times using the following settings:

- Training: Fold1,Validation: Fold2, Test: Fold3
- Training: Fold2,Validation: Fold1, Test: Fold3
- Training: Fold3,Validation: Fold2, Test: Fold1

Note: This code executes KD as described in the paper, so, it is designed to work on PanNuke dataset. 

### Inference
To run infer on tiles use `run_infer.py`.
```
python infer.py  --images_path IMAGES_PATH
                 --weights_path WEIGHTS_PATH 
                 --save_path SAVE_PATH 
                 [--step STEP] 
                 [--ext EXT]
                 [--overlay OVERLAY]
```

Options:

```
--images_path           images path
--weights_path          weights path of HoVer-UNet
--save_path             path to save predictions
[--step]                step used to generate overllaped patches, default: 192
[--ext]                 images extension, default: png
[--overlay]             overlay predition on image, default: True
```
`run_infer.py` creates two folder `predictions` and `overlay`. It save predictions as a dict object in a pickle file in `predictions` folder, and overlays prediction over image saving it in `overlay` folder. 

#### Model weights
To use pre-trained **HoVer-UNet** download weights [here](https://zenodo.org/records/10101807?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU2M2JkZWYyLTgyNzgtNGM4OC05YjhkLWQwYjk1NGMyZGIxZiIsImRhdGEiOnt9LCJyYW5kb20iOiIwY2I1ZDAyZWEwODNmNTNmZGZmODM1Y2M4YTcyNGRmNSJ9.HiosnYbIK79xB-l1-CIiTi7I6yoEUd_ZVNLCmYU5qevjB7LfkZCDexqclhBQrDN1cekzNajIAa2kqjpt9kchIQ).


![](docs/example.png)



## Citation
This work is currently submitted to International Symposium on Biomedical Imaging (ISBI) 2024.

@article{tommasino2023hover,
  title={" HoVer-UNet": Accelerating HoVerNet with UNet-based multi-class nuclei segmentation via knowledge distillation},
  author={Tommasino, Cristian and Russo, Cristiano and Rinaldi, Antonio Maria and Ciompi, Francesco},
  journal={arXiv preprint arXiv:2311.12553},
  year={2023}
}

## References
<a id="1" href="https://doi.org/10.1016/j.media.2019.101563">[1]</a> 
Gamper, J., Alemi Koohbanani, N., Benet, K., Khuram, A., & Rajpoot, N. (2019). PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification. In Digital Pathology: 15th European Congress, ECDP 2019, Warwick, UK, April 10–13, 2019, Proceedings 15 (pp. 11-19). Springer International Publishing.

<a id="2" href="https://doi.org/10.1016/j.media.2019.101563">[2]</a>
Graham, S., Vu, Q. D., Raza, S. E. A., Azam, A., Tsang, Y. W., Kwak, J. T., & Rajpoot, N. (2019). Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images. Medical image analysis, 58, 101563.

This repository include some code taken from https://github.com/vqdang/hover_net.

## Authors
[Cristian Tommasino](https://github.com/cristian921)

## License

This project is licensed under the APACHE License 2.0 - see the [LICENSE](LICENSE) file for details. 

Note that the PanNuke dataset is licensed under [Attribution-NonCommercial-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-nc-sa/4.0/), therefore the derived weights for HoVer-UNet and pre-processed PanNuke dataset to train HoVer-UNet are also shared under the same license. Please consider the implications of using the weights under this license on your work, and it's licensing. 
