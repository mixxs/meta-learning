# -*- coding: utf-8 -*-

import os
import shutil

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data.convert_dir_architecture import get_and_convert_dataset


def get_data_from_label(root_path: str, label: int):
    """
    根据label的值，从 路径 root_path/images/label 获取图像
    :param root_path: 数据集根目录
    :param label: 标签
    :return: 图像list，标签list（都是label）
    """
    images_path = os.path.join(root_path, "images")
    dir_name = str(label)
    dir_path = os.path.join(images_path, dir_name)
    images_list = []
    label_list = []
    for im_name in tqdm(os.listdir(dir_path), desc="正在读取图像", unit="image", ncols=100):
        im_path = os.path.join(dir_path, im_name)
        image = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        image = np.array([image])
        images_list.append(image)  # image形状是（28,28）
        label_list.append(label)
    return images_list, label_list  # （n,28,28）,(n,)


def get_split_datas(root_path: str, train_label_list: list, support_label_list: list):
    """
    获取划分后的数据+标签
    :param root_path: 数据集根目录
    :param train_label_list: 要放在training set中的类别
    :param support_label_list: 要放在support set 中的类别
    :return: 训练集数据list，训练集标签list，支持集数据list，支持集标签list
    """
    label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for label in tqdm(train_label_list, desc="正在检查输入的参数train_label_list", ncols=80, unit="item"):
        if label not in label_list:
            print(f"label必须是在[0,1,2,3,4,5,6,7,8,9]中，{label}不满足条件，非法输入程序即将退出")
            exit(-1)
    for label in tqdm(support_label_list, desc="正在检查输入的参数support_label_list", ncols=80, unit="itme"):
        if label not in label_list:
            print("label必须是在[0,1,2,3,4,5,6,7,8,9]中，{label}不满足条件，非法输入程序即将退出")
            exit(-1)
        if label in train_label_list:
            print("support set中的类别不可以出现在train set中，这不符合元学习few-shot的目标，程序即将退出")
            exit(-1)
    print("检查并调整目录结构")
    images_path = os.path.join(root_path, "images")
    MNIST_path = os.path.join(root_path, "MNIST")
    if not os.path.exists(images_path):
        if os.path.exists(MNIST_path):
            shutil.rmtree(MNIST_path)
        get_and_convert_dataset(root_path, True)
    train_images_list = []
    train_labels_ls = []
    support_images_list = []
    support_labels_ls = []
    for label in train_label_list:
        image_list, label_list = get_data_from_label(root_path, label)
        train_images_list += image_list
        train_labels_ls += label_list
    for label in support_label_list:
        image_list, label_list = get_data_from_label(root_path, label)
        support_images_list += image_list
        support_labels_ls += label_list

    return train_images_list, train_labels_ls, support_images_list, support_labels_ls


class BaseDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index):
        """

        :param index: 索引
        :return: image,label
        """
        return torch.tensor(self.datas[index], dtype=torch.float32), self.labels[index]

    def __len__(self):
        return len(self.datas)


class TripletDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index):
        anchor = self.datas[index]

        anchor_label = self.labels[index]
        negative_indices = np.where(np.array(self.labels) != anchor_label)[0]
        negative_index = np.random.choice(negative_indices)
        negative = self.datas[negative_index]

        positive_indices = np.where(np.array(self.labels) == anchor_label)[0]
        positive_indices = positive_indices[positive_indices != index]
        positive_index = np.random.choice(positive_indices)
        positive = self.datas[positive_index]

        return (torch.tensor(anchor, dtype=torch.float32), torch.tensor(positive, dtype=torch.float32),
                torch.tensor(negative, dtype=torch.float32)), False

    def __len__(self):
        return len(self.datas)


class BiDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index):
        image1 = self.datas[index]
        label1 = self.labels[index]

        # 以 0.5 的概率生成一个正样本或负样本
        should_get_same_class = np.random.randint(0, 2)
        if should_get_same_class:
            positive_indices = np.where(np.array(self.labels) == label1)[0]
            positive_indices = positive_indices[positive_indices != index]
            index2 = np.random.choice(positive_indices)
        else:
            negative_indices = np.where(np.array(self.labels) != label1)[0]
            index2 = np.random.choice(negative_indices)

        image2 = self.datas[index2]
        if should_get_same_class:
            label = 1
        else:
            label = 0
        return (torch.tensor(image1, dtype=torch.float32), torch.tensor(image2, dtype=torch.float32)), label

    def __len__(self):
        return len(self.datas)
