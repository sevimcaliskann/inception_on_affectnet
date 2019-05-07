from __future__ import print_function
from __future__ import division
import numpy as np
import torchvision
from torchvision import transforms
import time
import os
import copy
from model import ResNet_Train
from data.MoodDataset import MoodDataset
import pickle


def main():
    print('BEGINING')
    trainer = ResNet_Train()
    img_dir = os.path.join(trainer._opt.data_dir, trainer._opt.train_images_folder)
    list_path = trainer._opt.train_ids_file


    list_of_images = MoodDataset._read_ids(list_path)
    transform = transforms.Compose([transforms.Resize(size=(trainer._opt.image_size, trainer._opt.image_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                               std=[0.5, 0.5, 0.5])
                                          ])

    moods = trainer.get_last_fc_single_image(img_dir, list_of_images, transform)
    pickle.dump( moods, open( "/srv/glusterfs/csevim/datasets/affectnet/train_latent_%s.pkl" % trainer._opt.model, "wb" ) )
    print('END!')




if __name__ == '__main__':
    main()
