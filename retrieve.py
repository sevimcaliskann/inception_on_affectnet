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

    moods = trainer.infer_list_of_images(img_dir, list_of_images, transform)
    pickle.dump( moods, open( "/scratch_net/zinc/csevim/log/train_%s.pkl" % trainer._opt.model, "wb" ) )
    mean_error, std_error, l = trainer.check_sqr_mean_distance(moods, is_test = False)
    print('mean error: ', mean_error)
    print('std error: ', std_error)
    print('length of intersection: ', l)
    print('original length: ', len(list_of_images))
    file = open('log_%s.txt' % trainer._opt.model, 'wb')
    file.write('mean_error: \n')
    file.write(str(mean_error) + '\n')
    file.write('std_error: \n')
    file.write(str(std_error) + '\n')
    file.write('length of intersection: \n')
    file.write(str(l) + '\n')
    file.write('original length: \n')
    file.write(str(len(list_of_images)) + '\n')
    file.close()
    print('END!')




if __name__ == '__main__':
    main()
