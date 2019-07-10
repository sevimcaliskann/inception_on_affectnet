import os.path
import torchvision.transforms as transforms
from .dataset import DatasetBase
#from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import pickle
import cv2


class MoodDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(MoodDataset, self).__init__(opt, is_for_train)
        self._name = 'MoodDataset'
        self._read_dataset_paths()

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        # start_time = time.time()
        img = None
        emo = None
        mood = None
        while img is None or mood is None or emo is None:
            # if sample randomly: overwrite index
            if not self._opt.serial_batches:
                index = random.randint(0, self._dataset_size - 1)

            # get sample data
            sample_id = self._ids[index]

            img, img_path = self._get_img_by_id(sample_id)
            emo = self._get_cond_by_id(sample_id)
            mood = self._get_mood_by_id(sample_id)

            if img is None:
                print 'error reading image %s, skipping sample' % os.path.join(self._imgs_dir, sample_id)
            if emo is None:
                print 'error reading emo %s, skipping sample' % sample_id
            if mood is None:
                print 'error reading mood %s, skipping sample' % sample_id



        # transform data
        img = self._transform(Image.fromarray(img))

        # pack data
        sample = {'img': img,
                  'mood': mood,
                  'emo': emo,
                  'sample_id': sample_id,
                  'img_path': img_path
                  }

        # print (time.time() - start_time)

        return sample


    def __len__(self):
        return self._dataset_size


    @staticmethod
    def _read_ids(file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        return ids
        #return [id[:-4] for id in ids]

    def _read_dataset_paths(self):
        self._root = self._opt.data_dir
        self._imgs_dir = os.path.join(self._root, self._opt.train_images_folder) if self._is_for_train else os.path.join(self._root, self._opt.test_images_folder)
        info_filepath = self._opt.affectnet_info_file
        use_ids_filepath = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file

        # read ids
        self._ids = self._read_ids(use_ids_filepath)
        self._moods, self._emos = self._read_info(info_filepath)
        self._ids = list(set(self._ids).intersection(set(self._emos.keys())))
        print('#data: ', len(self._ids))

        # dataset size
        self._dataset_size = len(self._ids)

    def _create_transform(self):
        if self._is_for_train:
            transform_list = [transforms.Resize(size=(self._opt.image_size, self._opt.image_size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                              ]
        else:
            transform_list = [transforms.Resize(size=(self._opt.image_size, self._opt.image_size)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                              ]
        self._transform = transforms.Compose(transform_list)

    def _read_info(self, file_path):
        ids = np.loadtxt(file_path, delimiter = '\n', dtype = np.str)
        ids = ids[1:]
        cols = np.array([id.split(';') for id in ids])
        names = cols[:, 0]
        names = [name.split('/')[1] for name in names]
        names = [name.split(',')[0] for name in names]

	emos = np.array([row[-1].split(',')[1] for row in cols], dtype = np.int32)
        emos_dict = dict(zip(names, emos))

        cols = cols[:, -1]
        mood = [col.split(',')[-2:] for col in cols]
        mood_dict = dict(zip(names, mood))

        keys = set(self._ids).intersection(set(mood_dict.keys()))
        mood_dict = {k:mood_dict[k] for k in keys}

        keys = set(self._ids).intersection(set(emos_dict.keys()))
        emos_dict = {k:emos_dict[k] for k in keys}
        return mood_dict, emos_dict

    def _get_cond_by_id(self, id):
        emo = self._get_emo_by_id(id)
        return emo

    def _get_emo_by_id(self, id):
        if id in self._emos.keys():
            cond = self._emos[id]
            return cond
        else:
            return None

    def _get_mood_by_id(self, id):
        if id in self._moods.keys():
            cond = np.array(self._moods[id], dtype = np.float32)
            cond = np.array(cond)
            return cond
        else:
            return None

    def _get_img_by_id(self, id):
        filepath = os.path.join(self._imgs_dir, id)
        #filepath = os.path.join(self._imgs_dir, id+'_aligned')
        #filepath = os.path.join(filepath, 'face_det_000000.bmp')
        return self.read_cv2_img(filepath), filepath


    @staticmethod
    def read_cv2_img(path):
        '''
        Read color images
        :param path: Path to image
        :return: Only returns color images
        '''
        img = cv2.imread(path, -1)

        if img is not None:
            if len(img.shape) != 3:
                img = np.stack((img,)*3, axis=-1)
                #return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img
