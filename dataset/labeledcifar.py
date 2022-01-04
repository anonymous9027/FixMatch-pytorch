from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torchvision import transforms
import itertools
from torch.utils.data.sampler import Sampler
from .cifar import TransformFixMatch


class LABELEDCIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, root, seenclass_number, labeled_ratio, labeled=True, rand_number=0, transform=None,
                 target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(LABELEDCIFAR100, self).__init__(
            root, True, transform, target_transform, download)

        downloaded_list = self.train_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # labeled_classes = range(50)
        # labeled_ratio = 0.5
        labeled_classes = range(seenclass_number)
        np.random.seed(rand_number)

        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(
                labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
        else:
            self.shrink_data(unlabeled_idxs)

    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class LABELEDCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, seenclass_number, labeled_ratio, labeled=True, rand_number=0, transform=None,
                 target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(LABELEDCIFAR10, self).__init__(
            root, True, transform, target_transform, download)

        downloaded_list = self.train_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # labeled_classes = [0, 1, 2, 3, 4]
        # labeled_ratio = 0.5
        labeled_classes = range(seenclass_number)
        np.random.seed(rand_number)

        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(
                labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
        else:
            self.shrink_data(unlabeled_idxs)

    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class TransformThrice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        out3 = self.transform(inp)
        return out1, out2, out3


# Dictionary of transforms
dict_transform = {
    'cifar_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]),
    'cifar_train_simclr': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]),
    'cifar_train_justflip': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]),
    'cifar_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]),
    'stl10_train': transforms.Compose([
        transforms.Pad(12),
        transforms.RandomCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.44671062, 0.43980984, 0.40664645),
                             (0.26034098, 0.25657727, 0.27126738)),
    ]),
    'stl10_train_justflip': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.44671062, 0.43980984, 0.40664645),
                             (0.26034098, 0.25657727, 0.27126738)),
    ]),
    'stl10_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.44671062, 0.43980984, 0.40664645),
                             (0.26034098, 0.25657727, 0.27126738)),
    ]),
    'reuters': transforms.Compose([
        transforms.ToTensor(),
    ])
}


def GetCifar(dataset, seenclass_number, labeled_ratio,train_transform=-1):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    normal_mean = (0.5, 0.5, 0.5)
    normal_std = (0.5, 0.5, 0.5)

    if dataset == 'cifar10':    
        if train_transform ==-1:
            train_transform = TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
        else :
            train_transform = dict_transform['cifar_test']
        train_label_set = LABELEDCIFAR10(root='./data', seenclass_number=seenclass_number, labeled_ratio=labeled_ratio,
                                         labeled=True, download=True,
                                         transform=train_transform)
        train_unlabel_set = LABELEDCIFAR10(root='./data', seenclass_number=seenclass_number,
                                           labeled_ratio=labeled_ratio, labeled=False, download=True,
                                           transform=train_transform ,
                                           unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = LABELEDCIFAR10(root='./data', seenclass_number=seenclass_number, labeled_ratio=labeled_ratio,
                                  labeled=False, download=True,
                                  transform=dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
    elif dataset == 'cifar100':
        if train_transform ==-1:
            train_transform = TransformFixMatch(mean=cifar100_mean, std=cifar100_std)
        else :
            train_transform = dict_transform['cifar_test']
        train_label_set = LABELEDCIFAR100(root='./data', seenclass_number=seenclass_number, labeled_ratio=labeled_ratio,
                                          labeled=True, download=True,
                                          transform=train_transform)
        train_unlabel_set = LABELEDCIFAR100(root='./data', seenclass_number=seenclass_number,
                                            labeled_ratio=labeled_ratio, labeled=False, download=True,
                                            transform=train_transform,
                                            unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = LABELEDCIFAR100(root='./data', seenclass_number=seenclass_number, labeled_ratio=labeled_ratio,
                                   labeled=False, download=True,
                                   transform=dict_transform['cifar_test'],
                                   unlabeled_idxs=train_label_set.unlabeled_idxs)
    else:
        raise NotImplementedError
    return train_label_set, train_unlabel_set, test_set


if __name__ == '__main__':
    train_label_set = LABELEDCIFAR100(root='./data', seenclass_number=50, labeled_ratio=0.5, labeled=True,
                                      download=True,
                                      transform=TransformThrice(dict_transform['cifar_train']))
    train_unlabel_set = LABELEDCIFAR100(root='./data', seenclass_number=50, labeled_ratio=0.5, labeled=False,
                                        download=True,
                                        transform=TransformThrice(
                                            dict_transform['cifar_train']),
                                        unlabeled_idxs=train_label_set.unlabeled_idxs)
    test_set = LABELEDCIFAR100(root='./data', seenclass_number=50, labeled_ratio=0.5, labeled=False, download=True,
                               transform=dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
    import pdb

    pdb.set_trace()
