#!/usr/bin/env bash

python model.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--train_images_folder cropped2 \
--test_images_folder cropped2 \
--train_ids_file /srv/glusterfs/csevim/dataset_affectnet_analysis/train_small.csv \
--test_ids_file /srv/glusterfs/csevim/dataset_affectnet_analysis/test_mood.csv \
--affectnet_info_file /srv/glusterfs/csevim/datasets/affectnet/training.csv \
--name resnet50_5d \
--batch_size 64 \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotione/checkpoints \
--load_epoch -1 \
--dataset_mode mood \
--model resnet50
