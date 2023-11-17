# -*- coding: utf-8 -*-
import random
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import random_split, DataLoader, Subset

from data.BaseDataset import BaseDataset, BiDataset, TripletDataset, get_split_datas
from loss.Loss import TripletLoss, ContrastiveLoss
from model.getFeature import LeNet5
from model.Siamese import BiSiamese, TriSiamese


def get_config(cfg: str):
    """
    加载yaml配置文件的信息为dict
    :param cfg: yaml文件的位置
    :return: 配置dict
    """
    with open(cfg, 'r') as file:
        # 使用yaml.load函数加载YAML文件内容
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_data


def get_model(config: dict):
    """
    加载模型并冻结预训练权重
    :param config: 配置字典
    :return: 生成好的模型
    """
    print(config)
    classifier = config['classifier']
    if classifier == "LeNet":
        model = LeNet5(True)
    elif classifier == "BiSiamese":
        model = BiSiamese()
    else:
        model = TriSiamese()
    model = load_and_freeze(config, model)
    return model


def load_and_freeze(config: dict, model: BiSiamese):
    """
    加载预训练参数，并冻结部分参数
    :param config: 配置dict
    :param model: 修改之前的模型
    :return: 修改后的模型
    """
    # 加载
    if config["get_feature_pretrained"]:
        model.get_feature.load_state_dict(torch.load(config["get_feature_path"]))
        print("加载特征提取器权重")
    if config["Siamese_path"] is not None:
        # 此时model有一个模块还没有生成，必须调用前向传播
        images = (torch.full(size=(2, 1, 28, 28), fill_value=0.0, dtype=torch.float32),
                  torch.full(size=(2, 1, 28, 28), fill_value=0.0, dtype=torch.float32))
        model.forward(images)
        model.load_state_dict(torch.load(config["Siamese_path"]))
        print("加载模型权重")
    # 冻结
    if config["freeze_param"] is not None:
        if config["freeze_param"] == "get_feature":
            for param in model.get_feature.parameters():
                param.requires_grad = False
            print("冻结特征提取器参数")
        else:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.get_feature():
                param.requires_grad = True
            print("冻结孪生网络除特征提取器外的参数")
    else:
        for param in model.parameters():
            param.requires_grad = True
    return model


from typing import List, Tuple


def my_split_dataset(datas: List, labels: List, sizes: List[int]) -> List[Tuple[List, List]]:
    # 检查输入的sizes是否合理
    if sum(sizes) != len(datas):
        raise ValueError("Sum of sizes must equal the total number of datas")

    # 获取数据的随机排列
    indices = np.random.permutation(len(datas))

    # 创建结果列表
    result = []

    # 初始索引设为0
    start_idx = 0

    for size in sizes:
        # 提取索引
        end_idx = start_idx + size
        idxs = indices[start_idx:end_idx]

        # 根据索引提取数据和标签
        sub_datas = [datas[i] for i in idxs]
        sub_labels = [labels[i] for i in idxs]

        # 将子数据和子标签添加到结果中
        result.append((sub_datas, sub_labels))

        # 更新开始索引
        start_idx = end_idx

    return result


def get_dataset(cfg: dict):
    """
    根据配置字典生成训练集和测试集
    :param cfg: 配置字典
    :return: 训练集，测试集
    """
    train_label_list = cfg['train_set_class']
    support_label_list = cfg['support_set_class']
    root_path = cfg["dataset_path"]
    train_datas, train_labels, support_datas, support_labels = get_split_datas(root_path, train_label_list,
                                                                               support_label_list)
    if cfg['train_or_support'] == "train":
        if cfg["train_num"] < 1:
            train_size = int(len(train_labels) * cfg["train_num"])
            val_size = len(train_labels) - train_size
            size_list = [train_size, val_size]
            ((sp_train_datas, sp_train_labels), (sp_val_datas, sp_val_labels)) = my_split_dataset(train_datas,
                                                                                                  train_labels,
                                                                                                  size_list)
            if cfg["dataset_type"] == "BaseDataset":
                train_dataset, val_dataset = BaseDataset(sp_train_datas, sp_train_labels), BaseDataset(sp_val_datas,
                                                                                                       sp_val_labels)
            elif cfg["dataset_type"] == "BiDataset":
                train_dataset, val_dataset = BiDataset(sp_train_datas, sp_train_labels), BiDataset(sp_val_datas,
                                                                                                   sp_val_labels)
            else:
                train_dataset, val_dataset = TripletDataset(sp_train_datas, sp_train_labels), TripletDataset(
                    sp_val_datas, sp_val_labels)
            return train_dataset, val_dataset
        else:
            print("训练集的比例应该<1")
            exit(-1)

    if cfg["train_num"] < 1:
        print("不能少于一个数据")
        exit(-1)

    support_datas, support_labels, query_datas, query_labels = get_support_query_datas(support_datas,
                                                                                       support_labels,
                                                                                       support_label_list,
                                                                                       int(cfg["train_num"]))
    if cfg['dataset_type'] == "BiDataset":
        support_dataset = BiDataset(support_datas, support_labels)
        query_dataset = BiDataset(query_datas, query_labels)
    else:
        support_dataset = TripletDataset(support_datas, support_labels)
        query_dataset = TripletDataset(query_datas, query_labels)
    return support_dataset, query_dataset


def get_support_query_datas(datas, labels, label_list, num):
    """
    获取support set和query set
    :param datas: 所有数据
    :param labels: 所有标签
    :param label_list: support set中的标签组成的列表
    :param num: support set中，每一个类别的数据量
    :return:
    """
    # 首先根据标签将数据集分成各个类别
    class_to_data = defaultdict(list)
    for data, label in zip(datas, labels):
        if label in label_list:
            class_to_data[label].append(data)

    # 然后从每个类别中随机选择num个样本
    support_set = []
    support_labels = []
    query_set = []
    query_labels = []

    for label in label_list:
        random.shuffle(class_to_data[label])
        support_set.extend(class_to_data[label][:num])
        support_labels.extend([label] * num)

        query_set.extend(class_to_data[label][num:])
        query_labels.extend([label] * len(class_to_data[label][num:]))

    return support_set, support_labels, query_set, query_labels


def get_dataLoader(cfg: dict, dataset):
    """
    根据配置dict获取dataLoader
    :param cfg: 配置dict
    :param dataset: 数据集对象
    :return: dataLoader对象
    """
    batch_size = cfg['batch_size']
    shuffle = cfg['shuffle']
    dataset = dataset
    workers = cfg['workers']
    skip = cfg['skip_last']
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=skip)


def get_train_cfg(cfg: dict, params):
    """
    根据配置，生成优化器，损失函数，迭代次数，设备
    :param cfg: 配置dict
    :param params: 要训练的参数
    :return: 优化器，损失函数，迭代次数，设备
    """
    optim = cfg['optimizer']
    loss_function = cfg['loss']
    lr = cfg['lr']  # float
    epoch_total = cfg['epochs']  # int

    device = cfg['device']
    device = torch.device(device)

    if loss_function == 'CrossEntropyLoss':
        loss = nn.CrossEntropyLoss()
    elif loss_function == "TripletMarginLoss":
        loss = nn.TripletMarginLoss()
    elif loss_function == "BCELoss":
        loss = nn.BCELoss()
    elif loss_function == "MSELoss":
        loss = nn.MSELoss()
    else:
        loss = ContrastiveLoss()

    if optim == "RMSProp":
        optimizer = torch.optim.RMSprop(params, lr)
    elif optim == "Adam":
        optimizer = torch.optim.Adam(params, lr)
    else:
        optimizer = torch.optim.SGD(params, lr)
    return optimizer, loss, epoch_total, device
