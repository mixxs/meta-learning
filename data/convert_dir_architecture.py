# -*- coding: utf-8 -*-

import torch
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import transforms
import os
import cv2
from tqdm import tqdm


def merge_dataset(train_dataset: MNIST, test_dataset: MNIST):
    datas = torch.cat((train_dataset.data, test_dataset.data))
    labels = torch.cat((train_dataset.targets, test_dataset.targets))
    return datas, labels


def get_dataset(path: str, download: bool = False):
    train_dataset = MNIST(path, download=download, transform=transforms.ToTensor())
    test_dataset = MNIST(path, train=False, transform=transforms.ToTensor())
    print(f'Dataset：MNIST')
    print('===============================================================')
    print(f'Number of data：{len(train_dataset) + len(test_dataset)}')
    print(f'Shape of data：{train_dataset.data[0].shape}')
    print(f'Number of classes：{train_dataset.classes}')
    return merge_dataset(train_dataset, test_dataset)


def convert(datas: torch.Tensor, labels: torch.Tensor, root: str):
    MNIST_path = root
    images_path = os.path.join(MNIST_path, "images")
    for i in tqdm(range(10), desc="正在检查目录结构", ncols=100, unit="dir"):
        if not os.path.exists(os.path.join(images_path, f"{i}")):
            os.makedirs(os.path.join(images_path, f"{i}"))
    for i in tqdm(range(len(datas)), desc="正在分类写入图像", ncols=100, unit="image"):
        # 获取对应的路径
        dir_name = str(int(labels[i]))
        dir_path = os.path.join(images_path, dir_name)
        im_name = str(len(os.listdir(dir_path))) + ".jpg"
        im_path = os.path.join(dir_path, im_name)
        if os.path.exists(im_path):
            continue
        # 获取opencv格式的数据
        np_data = datas[i].numpy()  # 由于只有一个通道，所以不需要进行转置
        cv2.imwrite(im_path, np_data)


def get_and_convert_dataset(dataset_path, download):
    datas, labels = get_dataset(dataset_path, download)
    convert(datas, labels, dataset_path)


if __name__ == "__main__":
    dataset_root_path = "../dataset"
    download_from_web = False
    get_and_convert_dataset(dataset_root_path, download_from_web)
