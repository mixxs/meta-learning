import numpy as np
import torch
from tqdm import tqdm

from cfg.config import get_dataset
from data.BaseDataset import BaseDataset
from utils.tools import visualize_outputs, get_all_datas, select_random_indices
from model.getFeature import LeNet5
from model.Siamese import BiSiamese, TriSiamese

"""

"""
if __name__ == "__main__":
    # 为了兼容不同特征提取器的输出，孪生网络部分的第一层网络必须通过特征提取器输出形状进行构建。
    # tool_images用于进行一次前向传播，让孪生网络确定特征提取器的输出形状，从而生成自己的形状

    tool_images = (torch.full(size=(2, 1, 28, 28), fill_value=0.0, dtype=torch.float32),
                   torch.full(size=(2, 1, 28, 28), fill_value=0.0, dtype=torch.float32))

    # 没有经过Fine-Tune，只进行了预训练的特征提取网络
    get_feature1 = LeNet5(False)
    get_feature1.load_state_dict(torch.load("./utils/runs/LeNet_CrossEntropyLoss/run0/best_LeNet_CrossEntropyLoss.pt"))

    # 使用Triplet Loss进行Fine-Tune的Siamese network中的特征提取网络
    Tri_TriSiamese = TriSiamese()
    Tri_TriSiamese.forward(tool_images)
    Tri_TriSiamese.load_state_dict(torch.load("./utils/runs/TriSiamese_TripletMarginLoss/run0/best_TriSiamese_TripletMarginLoss.pt"))
    get_feature2 = Tri_TriSiamese.get_feature

    # 使用Contrastive Loss进行Fine-Tune的Siamese network中的特征提取网络
    Con_TriSiamese = TriSiamese()
    Con_TriSiamese.forward(tool_images)
    Con_TriSiamese.load_state_dict(
        torch.load("./utils/runs/TriSiamese_ContrastiveLoss/run0/best_TriSiamese_ContrastiveLoss.pt"))
    get_feature3 = Con_TriSiamese.get_feature

    # 在预训练特征提取器后，有用training set对自定义的、能直接是否相似的孪生网络参数进行进一步预训练，然后用CrossEntropy Loss进行Fine-Tune
    # 该孪生网络和常见的孪生网络不同的是，该网络输出只有两维，分别代表 相似 和 不相似。
    # 由于该孪生网络除特征提取器外还有很多参数，所以我对其整个网络在training set上进行了进一步的预训练。
    Bi_Siamese = BiSiamese()
    Bi_Siamese.forward(tool_images)
    Bi_Siamese.load_state_dict(torch.load("./utils/runs/BiSiamese_CrossEntropyLoss/run1/best_BiSiamese_CrossEntropyLoss.pt"))
    get_feature4 = Bi_Siamese.get_feature

    # 随机采样1%的数据
    all_data, all_label = get_all_datas("./dataset")
    all_data, all_label = np.array(all_data), np.array(all_label)
    sam_index = select_random_indices(all_label)
    sam_datas, sam_labels = all_data[sam_index], all_label[sam_index]
    print("————————————————————————图像获取完成—————————————————————————————")
    # 生成BaseDataset
    dataset = BaseDataset(sam_datas, sam_labels)

    # 生成dataloader
    from torch.utils.data import DataLoader

    dataLoader = DataLoader(dataset, 40, False, num_workers=6, drop_last=True)

    outs1ls = []
    outs2ls = []
    outs3ls = []
    outs4ls = []
    label_ls = []
    print("————————————————————————准备获取模型输出————————————————————————")
    for datas, labes in tqdm(dataLoader, desc="正在获取模型输出"):
        out1 = get_feature1(datas)
        out2 = get_feature2(datas)
        out3 = get_feature3(datas)
        out4 = get_feature4(datas)  # 40,10
        for i in range(out4.shape[0]):
            outs1ls.append(out1[i].cpu().detach().numpy())
            outs2ls.append(out2[i].cpu().detach().numpy())
            outs3ls.append(out3[i].cpu().detach().numpy())
            outs4ls.append(out4[i].cpu().detach().numpy())
            label_ls.append(labes[i].cpu().detach().numpy())

    outs1ls = np.array(outs1ls)
    outs2ls = np.array(outs2ls)
    outs3ls = np.array(outs3ls)
    outs4ls = np.array(outs4ls)

    visualize_outputs(outs1ls, label_ls)
    visualize_outputs(outs2ls, label_ls)
    visualize_outputs(outs3ls, label_ls)
    visualize_outputs(outs4ls, label_ls)
