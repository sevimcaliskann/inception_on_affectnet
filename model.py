from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from tensorboardX import SummaryWriter
from options import Options
from tqdm import tqdm
from data.MoodDataset import MoodDataset
from PIL import Image


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ResNet_Train():
    def __init__(self):
        self.name = 'ResNet'

        print("Initializing Datasets and Dataloaders...")

        self._opt = Options().parse()
        self.model, image_size = self.initialize_model(self._opt.model, 8, feature_extract=False, use_pretrained=False)
        self._opt.image_size = image_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self._opt.model != 'inception':
            layers = list(self.model.children())
            #layers = list(self.model.classifier.children())
            num_ftrs = layers[-1].in_features
            sub_net = layers[:-1]
            sub_net.append(Flatten())
            sub_net.append(nn.Linear(in_features=num_ftrs, out_features=3))
            sub_net.append(nn.Hardtanh())

            emo_layers = list()
            #emo_layers.append(nn.Linear(in_features=3, out_features=512))
            #emo_layers.append(nn.ReLU())
            emo_layers.append(nn.Linear(in_features=3, out_features=8))

            self.model = nn.Sequential(*sub_net)
            #self.model.classifier = nn.Sequential(*sub_net)
            self.emo_layer = nn.Sequential(*emo_layers)
            self.model = self.model.to(self.device)
            self.emo_layer = self.emo_layer.to(self.device)
            params = list(self.model.parameters()) + list(self.emo_layer.parameters())
            self.optimizer = optim.Adam(params, lr=self._opt.lr,
                                                 betas=[self._opt.adam_b1, self._opt.adam_b2])
        else:
            num_ftrs = self.model.AuxLogits.fc.in_features
            layers = list()
            layers.append(nn.Linear(in_features=num_ftrs, out_features=4))
            layers.append(nn.Hardtanh())
            self.model.AuxLogits.fc = nn.Sequential(*layers)

            emo_layers = list()
            emo_layers.append(nn.Linear(in_features=4, out_features=512))
            emo_layers.append(nn.ReLU())
            emo_layers.append(nn.Linear(in_features=512, out_features=8))
            self.aux_emo_layer = nn.Sequential(*emo_layers)


            num_ftrs = self.model.fc.in_features
            layers = list()
            layers.append(nn.Linear(in_features=num_ftrs, out_features=4))
            layers.append(nn.Hardtanh())
            self.model.fc = nn.Sequential(*layers)

            emo_layers = list()
            emo_layers.append(nn.Linear(in_features=4, out_features=512))
            emo_layers.append(nn.ReLU())
            emo_layers.append(nn.Linear(in_features=512, out_features=8))
            self.emo_layer = nn.Sequential(*emo_layers)

            self.model = self.model.to(self.device)
            self.emo_layer = self.emo_layer.to(self.device)
            self.aux_emo_layer = self.aux_emo_layer.to(self.device)
            params = list(self.model.parameters()) + list(self.emo_layer.parameters()) + list(self.aux_emo_layer.parameters())
            self.optimizer = optim.Adam(params, lr=self._opt.lr,
                                                 betas=[self._opt.adam_b1, self._opt.adam_b2])


        self.data_loader_train = CustomDatasetDataLoader(self._opt, is_for_train=True)
        self.data_loader_test = CustomDatasetDataLoader(self._opt, is_for_train=False)

        self.dataloaders_dict = {'train': self.data_loader_train.load_data(), 'val': self.data_loader_test.load_data()}

        self._dataset_train_size = len(self.data_loader_train)
        self._dataset_test_size = len(self.data_loader_test)
        print('#train images = %d' % self._dataset_train_size)
        print('#test images = %d' % self._dataset_test_size)

        # Detect if we have a GPU available

        self._Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self._Target = torch.cuda.LongTensor if torch.cuda.is_available() else torch.Tensor
        self._img = self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
        self._emo = self._Target(self._opt.batch_size, 1)
        self._mood = self._Tensor(self._opt.batch_size, 2)
        self._save_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        self._writer = SummaryWriter(self._save_dir)




        if self._opt.load_epoch>0:
            self.load()
        self.emo_criterion = nn.CrossEntropyLoss()
        self.mood_criterion = nn.L1Loss()
        #model = self.train_model(self.model, self.dataloaders_dict, self.criterion, self.optimizer, num_epochs=30, is_inception=False)
        #self._save_network(model, 31)

    def infer_single_image(self, filepath, transform, model):
        img = MoodDataset.read_cv2_img(filepath)
        if img is None:
            print('sample %s could not be read!' % filepath)
            return None
        img = torch.unsqueeze(transform(Image.fromarray(img)), 0)
        #self.model.eval()
        self._img.resize_(img.size()).copy_(img)
        outputs = model(self._img)
        outputs = outputs.cpu().detach().numpy()
        return outputs

    def get_last_fc_single_image(self, image_dir, list_of_images, transform):
        #model_child = nn.Sequential(*list(self.model.children())[:-1])
        #model_child.eval()
	self.model.eval()

        moods = dict()
        for img in tqdm(list_of_images):
            filepath = os.path.join(image_dir, img)
            outs = self.infer_single_image(filepath, transform, self.model)
            if outs is not None:
                moods[img] = np.squeeze(outs, axis=0)

        return moods




    def train_model(self, dataloaders, num_epochs=30, is_inception=False):
        since = time.time()

        #best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        number_iters_train = self._dataset_train_size//self._opt.batch_size
        number_iters_val = self._dataset_test_size//self._opt.batch_size

        for epoch in range(self._opt.load_epoch+1, num_epochs+1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                    self.emo_layer.train()
                else:
                    self.model.eval()   # Set model to evaluate mode
                    self.emo_layer.eval()

                running_loss = 0.0
                running_emo_loss = 0.0
                running_mood_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i_train_batch, train_batch in enumerate(dataloaders[phase]):
                    self._img.resize_(train_batch['img'].size()).copy_(train_batch['img'])
                    self._emo.resize_(train_batch['emo'].size()).copy_(train_batch['emo'])
                    self._mood.resize_(train_batch['mood'].size()).copy_(train_batch['mood'])

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase=='train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            moods, moods_aux = self.model(self._img)
                            emo = self.emo_layer(moods)
                            emo_aux = self.aux_emo_layer(moods_aux)

                            moods = moods.narrow(1, 1, 2)
                            moods_aux = moods_aux.narrow(1, 1, 2)
                            mood_loss = (self.mood_criterion(moods, self._mood) + self.mood_criterion(moods_aux, self._mood)*0.4)*10
                            emo_loss = self.emo_criterion(emo, self._emo) + self.emo_criterion(emo_aux, self._emo)*0.4
                            loss = mood_loss+emo_loss
                        else:
                            moods = self.model(self._img)
                            emo = self.emo_layer(moods)

                            moods = moods.narrow(1, 1, 2)
                            mood_loss = self.mood_criterion(moods, self._mood)*10
                            emo_loss = self.emo_criterion(emo, self._emo)
                            loss = mood_loss+emo_loss

                        _, preds = torch.max(emo, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    batch_loss = loss.item()
                    batch_emo_loss = emo_loss.item()
                    batch_mood_loss = mood_loss.item()
                    batch_corrects = torch.sum(preds == self._emo.data)


                    running_loss += batch_loss * self._opt.batch_size
                    running_emo_loss += batch_emo_loss * self._opt.batch_size
                    running_mood_loss += batch_mood_loss * self._opt.batch_size
                    running_corrects += batch_corrects

                    loss_dict = {'loss':batch_loss, 'acc':batch_corrects.double()/self._opt.batch_size, \
                                 'emo_loss':batch_emo_loss, 'mood_loss':batch_mood_loss}
                    if phase=='train':
                        self.plot_scalars(loss_dict, i_train_batch + number_iters_train*epoch, True)
                    else:
                        self.plot_scalars(loss_dict, i_train_batch + number_iters_val*epoch, False)




                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_emo_loss = running_emo_loss / len(dataloaders[phase])
                epoch_mood_loss = running_mood_loss / len(dataloaders[phase])
                epoch_acc = running_corrects.double() / len(dataloaders[phase])

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                print('{} Emo Loss: {:.4f} Mood Loss: {:.4f}'.format(phase, epoch_mood_loss, epoch_mood_loss))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                #    best_model_wts = copy.deepcopy(model.state_dict())
                #if phase == 'val':
                #    val_acc_history.append(epoch_acc)
            self.save(epoch)


        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        return model

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet18":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
        elif model_name == "resnet50":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "vgg11":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "vgg16":
            """ VGG16_bn
            """
            model_ft = models.vgg16_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224
        elif model_name == "vgg19":
            """ VGG19_bn
            """
            model_ft = models.vgg19_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224
        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs,num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size


    def save(self, label):
        # save networks
        self._save_network(label)
        self._save_optimizer(label)

    def load(self):
        load_epoch = self._opt.load_epoch
        self._load_network(load_epoch)
        self._load_optimizer(load_epoch)


    def _save_optimizer(self, epoch_label):
        save_filename = 'opt_epoch_%s.pth' % (epoch_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(self.optimizer.state_dict(), save_path)

    def _load_optimizer(self, epoch_label):
        load_filename = 'opt_epoch_%s.pth' % (epoch_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

        self.optimizer.load_state_dict(torch.load(load_path, map_location='cuda:0'))
        print('loaded optimizer: %s' % load_path)

    def _save_network(self, epoch_label):
        save_filename = 'net_epoch_%s.pth' % (epoch_label)
        save_path = os.path.join(self._save_dir, save_filename)
        if self._opt.model!='inception':
            torch.save({'model': self.model.state_dict(), 'emo_layer':self.emo_layer.state_dict()}, save_path)
        else:
            torch.save({'model': self.model.state_dict(),\
             'emo_layer':self.emo_layer.state_dict(), \
             'aux_emo_layer':self.aux_emo_layer.state_dict()}, save_path)
        print('saved net: %s' % save_path)

    def _load_network(self, epoch_label):
        load_filename = 'net_epoch_%s.pth' % (epoch_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path
        checkpoint = torch.load(load_path, map_location='cuda:0')
        if self._opt.model!='inception':
            self.model.load_state_dict(checkpoint['model'])
            self.emo_layer.load_state_dict(checkpoint['emo_layer'])
        else:
            self.model.load_state_dict(checkpoint['model'])
            self.emo_layer.load_state_dict(checkpoint['emo_layer'])
            self.aux_emo_layer.load_state_dict(checkpoint['aux_emo_layer'])
        print('loaded net: %s' % load_path)


    def plot_scalars(self, scalars, it, is_train):
        for label, scalar in scalars.items():
            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            self._writer.add_scalar(sum_name, scalar, it)

    def __del__(self):
        self._writer.close()


if __name__ == "__main__":
    trainer = ResNet_Train()
    model = trainer.train_model(trainer.dataloaders_dict, is_inception=trainer._opt.model=='inception')
