# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from cfg.config import get_dataLoader, get_model, get_train_cfg, get_dataset
from data.BaseDataset import BaseDataset, BiDataset, TripletDataset, get_data_from_label


def is_best(acc_list: list):
    lastest_acc = acc_list[-1]
    if lastest_acc == max(acc_list):
        return True
    else:
        return False


def get_saveDir(model_name: str, loss_name: str):
    runs_dir = "./runs"
    sub_dir_name = model_name + "_" + loss_name
    save_dir = os.path.join(runs_dir, sub_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    name = "run" + str(len(os.listdir(save_dir)))
    current_dir = os.path.join(save_dir, name)
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    return current_dir


def get_support_vector(model, label_list: list, support_set: Dataset):
    datas = support_set.datas
    labels = support_set.labels
    device = next(model.parameters()).device
    num = int(len(datas) / len(label_list))
    vec_list = []
    for label in label_list:
        vec_list.append(None)
        data_ba = []
        for data, la in zip(datas, labels):
            if la == label:
                data_ba.append(data)
        data_ba_ten = torch.tensor(data_ba, dtype=torch.float32).to(device)
        vecs = model(data_ba_ten)  # (1，num，84)
        vecs = vecs.squeeze()  # (num，84)
        vec = vecs.sum(dim=0)  # (84,)
        vec /= num
        index = label_list.index(label)
        vec_list[index] = vec

    return vec_list


def test(cfg, train_set, query, model):
    device = next(model.parameters()).device
    test_datas = query.datas
    test_labels = query.labels
    labels_list = cfg["support_set_class"]
    model.eval()
    correct = 0.0
    step = 0
    if cfg["classifier"] == "LeNet":  # 对LeNet进行测试
        dataset = BaseDataset(test_datas, test_labels)
        dataLoader = get_dataLoader(cfg, dataset)
        for st, ((inputs), labels) in enumerate(dataLoader):
            if st == 100:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)  # ((输出数，)batch size，vector length or class num)
            step += 1
            predicts = torch.argmax(outputs, dim=1)
            for i in range(len(predicts)):
                if predicts[i] == labels[i]:
                    correct += 1.0
    elif cfg["classifier"] == "TriSiamese":
        dataset = BaseDataset(test_datas, test_labels)
        dataLoader = get_dataLoader(cfg, dataset)
        for st, ((inputs), labels) in enumerate(dataLoader):
            if st == 100:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)  # ((输出数，)batch size，vector length or class num)
            step += 1
            support_vectors = get_support_vector(model, cfg["support_set_class"], train_set)
            outputs = torch.transpose(outputs, 0, 1)  # 调整为（batchsize, 输出数,vec len）
            for output, label in zip(outputs, labels):
                distances_list = []
                for vec in support_vectors:
                    distance = (torch.squeeze(vec) - torch.squeeze(output)).pow(2).sum(dim=-1)
                    distances_list.append(distance.detach().cpu().numpy())
                distances_list = np.array(distances_list)
                index = np.argmin(distances_list)
                predict = labels_list[index]
                if predict == label:
                    correct += 1.0
    else:
        dataset = BiDataset(test_datas, test_labels)
        dataLoader = get_dataLoader(cfg, dataset)
        for st, ((inputs), labels) in enumerate(dataLoader):
            if st == 100:
                break
            inputs = [inp.to(device) for inp in inputs]
            outputs = model.forward(inputs)  # ((输出数，)batch size，vector length or class num)
            step += 1
            for output, label in zip(outputs, labels):
                predict = torch.argmax(output)
                if predict == label:
                    correct += 1.0
    return correct / (float(step) * float(dataLoader.batch_size))


def train(cfg: dict):
    model = get_model(cfg)
    optm, criterion, epoch_total, device = get_train_cfg(cfg, model.parameters())
    loss_list = []
    acc_list = []
    model_name = cfg["classifier"]
    loss_name = cfg["loss"]
    save_dir = get_saveDir(model_name, loss_name)

    trainset, query = get_dataset(cfg)
    trainLoader = get_dataLoader(cfg, trainset)
    for epoch in tqdm(range(epoch_total), desc='正在训练模型', unit="epoch", ncols=100):
        model.train()
        model.to(device)
        epoch_loss = 0.0
        step = 0.0
        for _, (images, labels) in enumerate(trainLoader):
            optm.zero_grad()
            if type(images) == list:
                images = [image.to(device) for image in images]
            else:
                images = images.to(device)
            labels = labels.to(device)

            outputs = model.forward(images)
            if cfg["loss"] == "TripletMarginLoss":  # 需要转置
                outputs = torch.transpose(outputs, 0, 1)  # (batch size,输出数,vec len)
                loss = criterion(outputs[:, 0], outputs[:, 1], outputs[:, 2])
            elif cfg["loss"] == "ContrastiveLoss":  # 需要转置
                outputs = torch.transpose(outputs, 0, 1)  # (batch size,输出数,vec len)
                loss = criterion(outputs[:, 0], outputs[:, 1], labels)
            elif cfg["loss"] == "MSELoss":
                loss = criterion(outputs.squeeze(), labels.float())

            else:
                loss = criterion(outputs, labels)
            epoch_loss += loss
            loss.backward()
            optm.step()
            step += 1.0
        acc = test(cfg, trainset, query, model)
        epoch_loss /= (step * trainLoader.batch_size)
        print(f"\n第{epoch}个epoch: loss: {epoch_loss},acc: {acc}")
        loss_list.append(epoch_loss)
        acc_list.append(acc)
        if is_best(acc_list):
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_{model_name}_{loss_name}.pt'))
    torch.save(model.state_dict(), os.path.join(save_dir, f"last_{model_name}_{loss_name}.pt"))
    torch.save(loss_list, os.path.join(save_dir, f"{model_name}_{loss_name}_train_loss.pt"))
    torch.save(acc_list, os.path.join(save_dir, f"{model_name}_{loss_name}_train_acc.pt"))


def visualize_outputs(outputs, labels):
    '''
    outputs: np.array, the outputs of your model, shape is (n_samples, output_dim)
    labels: np.array or list, the label of each sample, shape is (n_samples,)
    '''

    # 使用TSNE进行降维
    tsne = TSNE(n_components=2, random_state=0)
    outputs_2d = tsne.fit_transform(outputs)

    # 获取各类别的颜色（这里使用了matplotlib的颜色循环）
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 绘制每个类别的点
    unique_labels = np.unique(labels)
    for i, label in tqdm(enumerate(unique_labels), desc="正在对输出进行可视化"):
        plt.scatter(outputs_2d[labels == label, 0], outputs_2d[labels == label, 1], color=colors[i % len(colors)],
                    label=label)

    plt.legend()
    plt.show()


def get_all_datas(root: str):
    images_list = []
    labels_list = []
    for i in range(10):
        images, labels = get_data_from_label(root, i)
        images_list += images
        labels_list += labels
    return images_list, labels_list  # (n,28,28),(n,)


def select_random_indices(input_list, percentage=0.01):
    num_total = len(input_list)
    num_select = int(num_total * percentage)
    indices = list(range(num_total))
    selected_indices = np.random.choice(indices, size=num_select, replace=False)
    return selected_indices.tolist()
