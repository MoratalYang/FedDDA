# This code is used to generate non-iid data with Feature Skew
import re
import sys, os

from torchvision.datasets import ImageFolder

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from collections import Counter


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        data (float): data.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath, label=0, domain=0, classname=""):
        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


def prepare_data_domain_partition_train(cfg, data_base_path):
    data_base_path = data_base_path
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    total_client_num = cfg.DATASET.USERS
    domain_name = cfg.DATASET.SOURCE_DOMAINS
    domain_num = len(domain_name)
    min_pic_require_size = 2
    domain_client_num = get_domain_client_num(domain_num, total_client_num)

    all_domain_trainset = []
    all_domain_testset = []
    global_test_set = []
    for domain_name_index in range(domain_num):
        current_domain_name = domain_name[domain_name_index]
        domain_n_clients = int(domain_client_num[domain_name_index])

        if cfg.DATASET.NAME == 'Office':
            global_domain_trainset = OfficeDataset(data_base_path, current_domain_name, transform=transform_train, train=True)
            global_domain_testset = OfficeDataset(data_base_path, current_domain_name, transform=transform_test, train=False)
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_office(data_base_path, current_domain_name,
                                                                                   cfg.DATASET.BETA,
                                                                                   n_parties=domain_n_clients,
                                                                                   min_require_size=min_pic_require_size)
        elif cfg.DATASET.NAME == 'DomainNet':
            global_domain_trainset = DomainNetDataset(data_base_path, current_domain_name, transform=transform_train, train=True)
            global_domain_testset = DomainNetDataset(data_base_path, current_domain_name, transform=transform_test, train=False)
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(global_domain_trainset, global_domain_testset
                                                                                   , beta=cfg.DATASET.BETA, K=100, n_parties=domain_n_clients,
                                                                                   min_require_size=min_pic_require_size)

        elif cfg.DATASET.NAME == 'PACS':
            global_domain_trainset = PACSDataset(data_base_path, current_domain_name, transform=transform_train, train=True)
            global_domain_testset = PACSDataset(data_base_path, current_domain_name, transform=transform_test, train=False)
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(global_domain_trainset, global_domain_testset
                                                                                   , beta=cfg.DATASET.BETA, K=7, n_parties=domain_n_clients,
                                                                                   min_require_size=min_pic_require_size)

        elif cfg.DATASET.NAME == 'Office31':
            global_domain_trainset = Office31Dataset(data_base_path, current_domain_name, transform=transform_train, train=True)
            global_domain_testset = Office31Dataset(data_base_path, current_domain_name, transform=transform_test, train=False)
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(global_domain_trainset, global_domain_testset
                                                                                   , beta=cfg.DATASET.BETA, K=len(global_domain_trainset.imagefolder_obj.classes),
                                                                                   n_parties=domain_n_clients,
                                                                                   min_require_size=min_pic_require_size)
        elif cfg.DATASET.NAME == 'OfficeHome':
            global_domain_trainset = OfficeHomeDataset(data_base_path, current_domain_name, transform=transform_train, train=True)
            global_domain_testset = OfficeHomeDataset(data_base_path, current_domain_name, transform=transform_test, train=False)
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(global_domain_trainset, global_domain_testset
                                                                                   , beta=cfg.DATASET.BETA, K=len(global_domain_trainset.imagefolder_obj.classes),
                                                                                   n_parties=domain_n_clients,
                                                                                   min_require_size=min_pic_require_size)

        if hasattr(global_domain_testset, 'imagefolder_obj'):

            classnames = global_domain_trainset.imagefolder_obj.classes
            lab2cname = {i: classnames[i] for i in range(len(classnames))}
        else:
            lab2cname = dict(zip(global_domain_testset.label, global_domain_testset.notation))
            classnames = [lab2cname[key] for key in sorted(lab2cname.keys())]

        global_domain_trainset = global_domain_trainset.data_detailed
        global_domain_testset = global_domain_testset.data_detailed

        domain_trainset = [[] for i in range(domain_n_clients)]
        domain_testset = [[] for i in range(domain_n_clients)]
        for i in range(domain_n_clients):
            if cfg.DATASET.NAME == 'Office':
                domain_trainset[i] = OfficeDataset(data_base_path, current_domain_name, net_dataidx_map_train[i], transform=transform_train)
                domain_testset[i] = OfficeDataset(data_base_path, current_domain_name, net_dataidx_map_test[i], train=False, transform=transform_test).data_detailed
            elif cfg.DATASET.NAME == 'DomainNet':
                domain_trainset[i] = DomainNetDataset(data_base_path, current_domain_name, net_dataidx_map_train[i], transform=transform_train)
                domain_testset[i] = DomainNetDataset(data_base_path, current_domain_name, net_dataidx_map_test[i], train=False,
                                                     transform=transform_test).data_detailed
            elif cfg.DATASET.NAME == 'PACS':
                domain_trainset[i] = PACSDataset(data_base_path, current_domain_name, net_dataidx_map_train[i], transform=transform_train)
                domain_testset[i] = PACSDataset(data_base_path, current_domain_name, net_dataidx_map_test[i], train=False,
                                                transform=transform_test).data_detailed

            elif cfg.DATASET.NAME == 'Office31':
                domain_trainset[i] = Office31Dataset(data_base_path, current_domain_name, net_dataidx_map_train[i], transform=transform_train)
                domain_testset[i] = Office31Dataset(data_base_path, current_domain_name, net_dataidx_map_test[i], train=False,
                                                    transform=transform_test).data_detailed

            elif cfg.DATASET.NAME == 'OfficeHome':
                domain_trainset[i] = OfficeHomeDataset(data_base_path, current_domain_name, net_dataidx_map_train[i], transform=transform_train)
                domain_testset[i] = OfficeHomeDataset(data_base_path, current_domain_name, net_dataidx_map_test[i], train=False,
                                                      transform=transform_test).data_detailed
            # lab2cname = domain_trainset[i].lab2cname
            # classnames = domain_trainset[i].classnames
            domain_trainset[i] = domain_trainset[i].data_detailed

        all_domain_trainset.append(domain_trainset)
        all_domain_testset.append(domain_testset)
        global_test_set.append(global_domain_testset)

    train_data_num_list = []
    test_data_num_list = []
    train_set = []
    test_set = []
    for dataset in all_domain_trainset:
        for i in range(len(dataset)):
            train_data_num_list.append(len(dataset[i]))
            train_set.append(dataset[i])
    for dataset in all_domain_testset:
        for i in range(len(dataset)):
            test_data_num_list.append(len(dataset[i]))
            test_set.append(dataset[i])
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    return train_set, test_set, global_test_set, classnames, lab2cname


def get_domain_client_num(domain_num, total_client_num):
    # if equal_dis:
    n_clients = total_client_num // domain_num
    not_allocated_num = total_client_num % domain_num

    domain_client_num = np.ones(domain_num) * n_clients
    remain_num = np.random.randint(domain_num, size=not_allocated_num)
    for i in range(len(remain_num)):
        domain_client_num[remain_num[i]] += 1

    return domain_client_num


class OfficeHomeDataset(Dataset):
    def __init__(self, base_path, site, net_dataidx_map=None, train=True, transform=None, subset_train_num=7, subset_capacity=10):
        # site = site.lower()
        self.base_path = base_path

        self.imagefolder_obj = ImageFolder(self.base_path + '/office_home/' + site + '/', transform)
        self.train = train
        all_data = self.imagefolder_obj.samples
        self.train_data_list = []
        self.test_data_list = []
        for i in range(len(all_data)):
            if i % subset_capacity <= subset_train_num:
                self.train_data_list.append(all_data[i])
            else:
                self.test_data_list.append(all_data[i])

        if train:
            data_list = np.array(self.train_data_list)
        else:
            data_list = np.array(self.test_data_list)

        self.site_domain = {'Art': 0, 'Clipart': 1, 'Product': 2, 'Real_World': 3}

        self.site = site

        self.imgs = data_list[:, 0]
        self.label = (data_list[:, 1]).astype('int32')

        if net_dataidx_map is not None:  # 如果有数据划分
            self.imgs = self.imgs[net_dataidx_map]
            self.label = self.label[net_dataidx_map]

        self.transform = transform
        self.data_detailed = self._convert()

    def __len__(self):
        return len(self.label)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.label)):
            img_path = self.imgs[i]
            data_idx = img_path
            target_idx = int(self.label[i])
            notation_idx = self.imagefolder_obj.classes[target_idx]
            item = Datum(impath=data_idx, label=int(target_idx), domain=self.site, classname=notation_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.label[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class Office31Dataset(Dataset):
    def __init__(self, base_path, site, net_dataidx_map=None, train=True, transform=None, subset_train_num=7, subset_capacity=10):
        site = site.lower()
        self.base_path = base_path

        self.imagefolder_obj = ImageFolder(self.base_path + '/OFFICE31/' + site + '/', transform)
        self.train = train
        all_data = self.imagefolder_obj.samples
        self.train_data_list = []
        self.test_data_list = []
        for i in range(len(all_data)):
            if i % subset_capacity <= subset_train_num:
                self.train_data_list.append(all_data[i])
            else:
                self.test_data_list.append(all_data[i])

        if train:
            data_list = np.array(self.train_data_list)
        else:
            data_list = np.array(self.test_data_list)

        self.site_domain = {'amazon': 0, 'dslr': 1, 'webcam': 2}

        self.site = site

        self.imgs = data_list[:, 0]
        self.label = (data_list[:, 1]).astype('int32')

        if net_dataidx_map is not None:  # 如果有数据划分
            self.imgs = self.imgs[net_dataidx_map]
            self.label = self.label[net_dataidx_map]

        self.transform = transform
        self.data_detailed = self._convert()

    def __len__(self):
        return len(self.label)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.label)):
            img_path = self.imgs[i]
            data_idx = img_path
            target_idx = int(self.label[i])
            notation_idx = self.imagefolder_obj.classes[target_idx]
            item = Datum(impath=data_idx, label=int(target_idx), domain=self.site, classname=notation_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.label[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, net_dataidx_map=None, train=True, transform=None):
        self.base_path = base_path
        if train:
            path = os.path.join(self.base_path, 'office_caltech_10/{}_train.pkl'.format(site))
            self.paths, self.notation = np.load(path, allow_pickle=True)
        else:
            path = os.path.join(self.base_path, 'office_caltech_10/{}_test.pkl'.format(site))
            self.paths, self.notation = np.load(path, allow_pickle=True)

        self.site_domain = {'amazon': 0, 'caltech': 1, 'dslr': 2, 'webcam': 3}
        self.domain = self.site_domain[site]
        self.lab2cname = {'back_pack': 0, 'bike': 1, 'calculator': 2, 'headphones': 3, 'keyboard': 4,
                          'laptop_computer': 5,
                          'monitor': 6, 'mouse': 7, 'mug': 8,
                          'projector': 9}
        self.classnames = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor',
                           'mouse',
                           'mug', 'projector']
        self.label = np.asarray([self.lab2cname[text] for text in self.notation])
        if train:
            print('Counter({}_train data:)'.format(site), Counter(self.label))
        else:
            print('Counter({}_test data:)'.format(site), Counter(self.label))

        if net_dataidx_map is not None:  # 如果有数据划分
            self.paths = self.paths[net_dataidx_map]
            self.label = self.label[net_dataidx_map]
            self.notation = self.notation[net_dataidx_map].tolist()
        self.transform = transform
        self.data_detailed = self._convert()

    def __len__(self):
        return len(self.label)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.label)):
            img_path = self.paths[i]
            data_idx = os.path.join(self.base_path, img_path)
            target_idx = self.label[i]
            notation_idx = self.notation[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=int(self.domain), classname=notation_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.label[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    

class PACSDataset(Dataset):
    def __init__(self, base_path, site, net_dataidx_map=None, train=True, transform=None):
        site = site.lower()
        self.base_path = base_path
        if train:
            self.split_file = os.path.join(self.base_path, f'PACS/label/{site}_train_kfold.txt')
        else:
            self.split_file = os.path.join(self.base_path, f'PACS/label/{site}_test_kfold.txt')
        self.imgs, self.notation, self.label = self.read_txt(self.split_file, self.base_path + '/PACS')

        self.site_domain = {'art_painting': 0, 'cartoon': 1, 'photo': 2, 'sketch': 3}

        self.site = site

        self.imgs = np.asarray(self.imgs)
        self.label = np.asarray(self.label) - 1
        self.notation = np.asarray(self.notation)
        if net_dataidx_map is not None:  # 如果有数据划分
            self.imgs = self.imgs[net_dataidx_map]
            self.label = self.label[net_dataidx_map]
            self.notation = self.notation[net_dataidx_map].tolist()
        self.transform = transform
        self.data_detailed = self._convert()

    def read_txt(self, txt_path, root_path):
        imgs = []
        notations = []
        targets = []
        with open(txt_path, 'r') as f:
            txt_component = f.readlines()
        for line_txt in txt_component:
            label_name = line_txt.split('/')[1]
            line_txt = line_txt.replace('\n', '')
            line_txt = line_txt.split(' ')
            imgs.append(os.path.join(root_path, 'raw_images', line_txt[0]))
            notations.append(label_name)
            targets.append(int(line_txt[1]))
        return imgs, notations, targets

    def __len__(self):
        return len(self.label)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.label)):
            img_path = self.imgs[i]
            data_idx = img_path
            target_idx = self.label[i]
            notation_idx = self.notation[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=self.site, classname=notation_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.label[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, net_dataidx_map=None, train=True, transform=None):
        site = site.lower()
        self.base_path = base_path
        if train:
            self.split_file = os.path.join(self.base_path, f'DomainNet/{site}_train.txt')
        else:
            self.split_file = os.path.join(self.base_path, f'DomainNet/{site}_test.txt')

        self.imgs, self.notation, self.label = DomainNetDataset.read_txt(self.split_file, self.base_path + '/DomainNet')

        self.site_domain = {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}

        self.site = site

        self.imgs = np.asarray(self.imgs)
        self.label = np.asarray(self.label)
        self.notation = np.asarray(self.notation)
        if net_dataidx_map is not None: 
            self.imgs = self.imgs[net_dataidx_map]
            self.label = self.label[net_dataidx_map]
            self.notation = self.notation[net_dataidx_map].tolist()
        self.transform = transform
        self.data_detailed = self._convert()

    @staticmethod
    def read_txt(txt_path, root_path):
        imgs = []
        notations = []
        targets = [] 
        with open(txt_path, 'r') as f:
            txt_component = f.readlines()
        for line_txt in txt_component:
            label_name = line_txt.split('/')[1]
            line_txt = line_txt.replace('\n', '')
            line_txt = line_txt.split(' ')
            imgs.append(os.path.join(root_path, line_txt[0]))
            notations.append(label_name)
            targets.append(int(line_txt[1]))
        return imgs, notations, targets

    def __len__(self):
        return len(self.label)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.label)):
            img_path = self.imgs[i]
            data_idx = img_path
            target_idx = self.label[i]
            notation_idx = self.notation[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=self.site, classname=notation_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.label[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


def Dataset_partition_domain(global_domain_trainset, global_domain_testset, beta, K, n_parties=5, min_require_size=2):
    min_size = 0

    train_path = global_domain_trainset.imgs
    train_labels = global_domain_trainset.label
    train_labels = np.array(train_labels)
    test_path = global_domain_testset.imgs
    test_labels = global_domain_testset.label
    test_labels = np.array(test_labels)

    N_train = len(train_labels)
    N_test = len(test_labels)
    net_dataidx_map_train = {}
    net_dataidx_map_test = {}

    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_parties)]
        idx_batch_test = [[] for _ in range(n_parties)]
        for k in range(K):
            train_idx_k = np.where(train_labels == k)[0]
            test_idx_k = np.where(test_labels == k)[0]
            train_idx_k = np.array(train_idx_k)
            test_idx_k = np.array(test_idx_k)
            np.random.seed(0)
            np.random.shuffle(train_idx_k)
            np.random.shuffle(test_idx_k)
            if beta == 0:
                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.array_split(train_idx_k, n_parties))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.array_split(test_idx_k, n_parties))]
            else:
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
                proportions = proportions / proportions.sum()
                proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
                train_part_list = np.split(train_idx_k, proportions_train)
                test_part_list = np.split(test_idx_k, proportions_test)
                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, train_part_list)]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, test_part_list)]

            min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
            min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
            min_size = min(min_size_test, min_size_train)

    for j in range(n_parties):
        np.random.shuffle(idx_batch_train[j])
        np.random.shuffle(idx_batch_test[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
        net_dataidx_map_test[j] = idx_batch_test[j]

    traindata_cls_counts = record_net_data_stats(train_labels, net_dataidx_map_train)
    print(global_domain_trainset.site, "Training data split: ", traindata_cls_counts)
    testdata_cls_counts = record_net_data_stats(test_labels, net_dataidx_map_test)
    print(global_domain_trainset.site, "Testing data split: ", testdata_cls_counts)
    return net_dataidx_map_train, net_dataidx_map_test


def Dataset_partition_office(base_path, site, beta, n_parties=3, min_require_size=2):
    min_size = 0
    K = 10

    train_path = os.path.join(base_path, 'office_caltech_10/{}_train.pkl'.format(site))
    test_path = os.path.join(base_path, 'office_caltech_10/{}_test.pkl'.format(site))
    _, train_text_labels = np.load(train_path, allow_pickle=True)
    _, test_text_labels = np.load(test_path, allow_pickle=True)

    label_dict = {'back_pack': 0, 'bike': 1, 'calculator': 2, 'headphones': 3, 'keyboard': 4, 'laptop_computer': 5, 'monitor': 6, 'mouse': 7, 'mug': 8, 'projector': 9}
    train_labels = np.asarray([label_dict[text] for text in train_text_labels])
    test_labels = np.asarray([label_dict[text] for text in test_text_labels])
    N_train = train_labels.shape[0]
    net_dataidx_map_train = {}
    net_dataidx_map_test = {}

    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_parties)]
        idx_batch_test = [[] for _ in range(n_parties)]
        for k in range(K):
            train_idx_k = np.where(train_labels == k)[0]
            test_idx_k = np.where(test_labels == k)[0]
            np.random.seed(0)
            np.random.shuffle(train_idx_k)
            np.random.shuffle(test_idx_k)
            if beta == 0:
                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.array_split(train_idx_k, n_parties))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.array_split(test_idx_k, n_parties))]
            else:
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
                proportions = proportions / proportions.sum()
                proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
                train_part_list = np.split(train_idx_k, proportions_train)
                test_part_list = np.split(test_idx_k, proportions_test)
                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, train_part_list)]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, test_part_list)]

            min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
            min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
            min_size = min(min_size_test, min_size_train)

    for j in range(n_parties):
        np.random.shuffle(idx_batch_train[j])
        np.random.shuffle(idx_batch_test[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
        net_dataidx_map_test[j] = idx_batch_test[j]

    traindata_cls_counts = record_net_data_stats(train_labels, net_dataidx_map_train)
    print(site, "Training data split: ", traindata_cls_counts)
    testdata_cls_counts = record_net_data_stats(test_labels, net_dataidx_map_test)
    print(site, "Testing data split: ", testdata_cls_counts)
    return net_dataidx_map_train, net_dataidx_map_test


class DomainNetDataset_sub(Dataset):
    def __init__(self, base_path, site, net_dataidx_map, train=True, transform=None):
        site = site.lower()
        self.base_path = base_path
        if train:
            self.split_file = os.path.join(self.base_path, f'DomainNet/{site}_train.txt')
        else:
            self.split_file = os.path.join(self.base_path, f'DomainNet/{site}_test.txt')

        self.imgs, self.notation, self.label = DomainNetDataset.read_txt(self.split_file, self.base_path + '/DomainNet')

        self.domain = site

        self.lab2cname = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5, 'whale': 6, 'windmill': 7, 'wine_glass': 8, 'zebra': 9}
        self.classnames = ['bird', 'feather', 'headphones', 'ice_cream', 'teapot', 'tiger', 'whale', 'windmill', 'wine_glass', 'zebra']

        if train:
            print('Counter({}_train data:)'.format(site), Counter(self.label))
        else:
            print('Counter({}_test data:)'.format(site), Counter(self.label))
        self.notation = self.notation
        self.transform = transform
        self.data_detailed = self._convert()

        self.imgs = np.array(self.imgs)[net_dataidx_map]
        self.label = np.array(self.label)[net_dataidx_map]
        self.notation = np.array(self.notation)[net_dataidx_map].tolist()

    @staticmethod
    def read_txt(txt_path, root_path):
        imgs = []
        notations = []  # 存储文字标签
        targets = []  # 存储数字标签
        with open(txt_path, 'r') as f:
            txt_component = f.readlines()
        for line_txt in txt_component:
            label_name = line_txt.split('/')[1]
            line_txt = line_txt.replace('\n', '')
            line_txt = line_txt.split(' ')
            imgs.append(os.path.join(root_path, line_txt[0]))
            notations.append(label_name)
            targets.append(int(line_txt[1]))
        return imgs, notations, targets

    def __len__(self):
        return len(self.label)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.label)):
            img_path = self.imgs[i]
            data_idx = img_path
            target_idx = self.label[i]
            notation_idx = self.notation[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=self.site, classname=notation_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.label[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
