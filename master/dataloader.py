from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from torch.autograd import Variable
# from .utils import download_url, check_integrity

class lunanod(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    def __init__(self, npypath, fnamelst, labellst, featlst, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_feat = featlst
            for label, fentry in zip(labellst, fnamelst):
                file = os.path.join(npypath, fentry)
                self.train_data.append(np.load(file))
                self.train_labels.append(label)
            self.train_data = np.concatenate(self.train_data)
            print(len(fnamelst))
            print(self.train_data.shape)
            self.train_data = self.train_data.reshape((len(fnamelst), 32, 32, 32))
            # self.train_labels = np.asarray(self.train_labels)
            # self.train_data = self.train_data.transpose((0, 2, 3, 4, 1))  # convert to HWZC
            self.train_len = len(fnamelst)
        else:
            self.test_data = []
            self.test_labels = []
            self.test_feat = featlst
            for label, fentry in zip(labellst, fnamelst):
                # if fentry.shape[0] != 32 or fentry.shape[1] != 32 or fentry.shape[2] != 32:
                #     print(fentry.shape, type(fentry), type(fentry)!='str')
                if not isinstance(fentry,str):
                    self.test_data.append(fentry)
                    self.test_labels.append(label)
                    # print('1')
                else:
                    file = os.path.join(npypath, fentry)
                    self.test_data.append(np.load(file))
                    self.test_labels.append(label)
            self.test_data = np.concatenate(self.test_data)
            # print(self.test_data.shape)
            self.test_data = self.test_data.reshape((len(fnamelst), 32, 32, 32))
            # self.test_labels = np.asarray(self.test_labels)
            # self.test_data = self.test_data.transpose((0, 2, 3, 4, 1))  # convert to HWZC
            self.test_len = len(fnamelst)
            print(self.test_data.shape, len(self.test_labels), len(self.test_feat))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target, feat = self.train_data[index], self.train_labels[index], self.train_feat[index]
        else:
            img, target, feat = self.test_data[index], self.test_labels[index], self.test_feat[index]
        # img = torch.from_numpy(img) 
        # img = img.cuda(async = True)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print('1', img.shape, type(img))
        # img = Image.fromarray(img)
        # print('2', img.size)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(img.shape, target.shape, feat.shape)
        # print(target)

        return img, target, feat

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.test_len


class lunanod_2(data.Dataset):

    def __init__(self, npypath, fnamelst, labellst, featlst, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_feat = featlst
            for label, fentry in zip(labellst, fnamelst):
                file = os.path.join(npypath, fentry)
                self.train_data.append(np.load(file))
                self.train_labels.append(label)
            self.train_data = np.concatenate(self.train_data)
            print(len(fnamelst))
            print(self.train_data.shape)
            self.train_data = self.train_data.reshape((len(fnamelst), 32, 32, 32))

            positive_data = [(d, l, f) for d, l, f in zip(self.train_data, self.train_labels, self.train_feat) if
                             l == 1]
            negative_data = [(d, l, f) for d, l, f in zip(self.train_data, self.train_labels, self.train_feat) if
                             l == 0]

            print(len(negative_data))
            print(len(positive_data))
            # if len(negative_data) > len(positive_data):
            #     # negative_data = random.sample(negative_data, len(positive_data))
            #     # negative_data = random.sample(negative_data, int(2 / 3 * len(positive_data)))
            #     negative_data = random.sample(negative_data, int(0.77 * len(positive_data)))

            # negative_data = random.sample(negative_data, int(0.77 * len(positive_data)))
            positive_data = random.choices(positive_data, k=len(negative_data))
            # positive_data = random.sample(positive_data, int(0.77 * len(negative_data)))
            print(len(negative_data))
            print(len(positive_data))
            self.train_data = [d for d, _, _ in positive_data] + [d for d, _, _ in negative_data]
            self.train_labels = [l for _, l, _ in positive_data] + [l for _, l, _ in negative_data]
            self.train_feat = [f for _, _, f in positive_data] + [f for _, _, f in negative_data]

            combined = list(zip(self.train_data, self.train_labels, self.train_feat))
            random.shuffle(combined)
            self.train_data, self.train_labels, self.train_feat = zip(*combined)

            self.train_len = len(self.train_data)

        else:
            self.test_data = []
            self.test_labels = []
            self.test_feat = featlst
            for label, fentry in zip(labellst, fnamelst):
                if not isinstance(fentry, str):
                    self.test_data.append(fentry)
                    self.test_labels.append(label)
                else:
                    file = os.path.join(npypath, fentry)
                    self.test_data.append(np.load(file))
                    self.test_labels.append(label)
            self.test_data = np.concatenate(self.test_data)
            self.test_data = self.test_data.reshape((len(fnamelst), 32, 32, 32))
            self.test_len = len(fnamelst)
            print(self.test_data.shape, len(self.test_labels), len(self.test_feat))

    def __getitem__(self, index):
        if self.train:
            img, target, feat = self.train_data[index], self.train_labels[index], self.train_feat[index]
        else:
            img, target, feat = self.test_data[index], self.test_labels[index], self.test_feat[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, feat

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.test_len
