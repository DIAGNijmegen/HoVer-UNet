#! /bin/bash
encoder="mit_b2"
alpha=0.5
t=3
base_path="/work/cristian/test_fasthovernet/data"
path_save="/work/cristian/test_fasthovernet"
pannuke_path="/work/oldWork/default-ubuntu-rinaldi2-work-pvc-215307e1-04be-4abb-8a29-26eb803ad2b3/cristian/data/Pannuke"
project_name="Test"
experiment_group=0
nr_epochs=5
use_true_labels=1
use_hovernet_predictions=1
batch_size=32

echo "nr_epochs "$nr_epochs

python3 train.py --base_project_dir $path_save \
--project_name  $project_name \
--experiment_group $experiment_group \
--experiment_id 0 \
--path_train $base_path/fold1.h5 \
--path_val $base_path/fold2.h5 \
--path_test $base_path/fold3.h5 \
--pannuke_path $pannuke_path \
--use_hovernet_predictions $use_hovernet_predictions \
--use_true_labels $use_true_labels \
--batch_size $batch_size \
--encoder $encoder \
--nr_epochs $nr_epochs \
--loss_alpha $alpha \
--loss_t $t \


python3 train.py --base_project_dir $path_save \
--project_name  $project_name \
--experiment_group $experiment_group \
--experiment_id 1 \
--path_train $base_path/fold2.h5 \
--path_val $base_path/fold1.h5 \
--path_test $base_path/fold3.h5 \
--pannuke_path $pannuke_path \
--use_hovernet_predictions $use_hovernet_predictions \
--use_true_labels $use_true_labels \
--batch_size $batch_size \
--encoder $encoder \
--nr_epochs $nr_epochs \
--loss_alpha $alpha \
--loss_t $t \


python3 train.py --base_project_dir $path_save \
--project_name  $project_name \
--experiment_group $experiment_group \
--experiment_id 2 \
--path_train $base_path/fold3.h5 \
--path_val $base_path/fold2.h5 \
--path_test $base_path/fold1.h5 \
--pannuke_path $pannuke_path \
--use_hovernet_predictions $use_hovernet_predictions \
--use_true_labels $use_true_labels \
--batch_size $batch_size \
--encoder $encoder \
--nr_epochs $nr_epochs \
--loss_alpha $alpha \
--loss_t $t \



