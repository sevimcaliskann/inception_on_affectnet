#!/usr/bin/env bash

export SGE_GPU_ALL="$(ls -rt /tmp/lock-gpu*/info.txt | xargs grep -h $(whoami) | awk '{print $2}' | paste -sd "," -)"
export SGE_GPU=$(echo $SGE_GPU_ALL |rev|cut -d, -f1|rev) # USE LAST GPU by request time.
echo "SGE gpu=$SGE_GPU allocated in this use"

CUDA_VISIBLE_DEVICES=$SGE_GPU python retrieve.py \
--data_dir /srv/glusterfs/csevim/datasets/affectnet \
--train_images_folder cropped2 \
--test_images_folder cropped2 \
--train_ids_file /scratch_net/zinc/csevim/dataset_affectnet_analysis/train_small.csv \
--test_ids_file /scratch_net/zinc/csevim/dataset_affectnet_analysis/test_mood.csv \
--affectnet_info_file /srv/glusterfs/csevim/datasets/affectnet/training.csv \
--name resnet_train_emo \
--batch_size 64 \
--checkpoints_dir /srv/glusterfs/csevim/datasets/emotione/checkpoints \
--load_epoch -1 \
--dataset_mode mood \
--model resnet18
