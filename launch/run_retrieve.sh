#!/usr/bin/env bash

python retrieve.py \
--data_dir /srv/beegfs02/scratch/emotion_perception/data/csevim/affwild/aff_wild_annotations_bboxes_landmarks_new \
--train_images_folder cropped \
--test_images_folder cropped \
--train_ids_file /srv/beegfs02/scratch/emotion_perception/data/csevim/affwild/aff_wild_annotations_bboxes_landmarks_new/train_affwild_images.csv \
--test_ids_file /srv/beegfs02/scratch/emotion_perception/data/csevim/affwild/aff_wild_annotations_bboxes_landmarks_new/test_affwild_images.csv \
--affectnet_info_file /srv/beegfs02/scratch/emotion_perception/data/csevim/datasets/affectnet/training.csv \
--name resnet50_mood_emo_1000 \
--batch_size 64 \
--checkpoints_dir /srv/beegfs02/scratch/emotion_perception/data/csevim/datasets/emotione/checkpoints \
--load_epoch -1 \
--dataset_mode mood \
--model resnet50
