import torch
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter(r'./logs')  # logs就是步骤1. 创建的文件夹

    loss_pretrain_LeNet = torch.load("./utils/runs/LeNet_CrossEntropyLoss/run0/LeNet_CrossEntropyLoss_train_loss.pt")
    acc_pretrain_LeNet = torch.load("./utils/runs/LeNet_CrossEntropyLoss/run0/LeNet_CrossEntropyLoss_train_acc.pt")

    loss_Triplet_TriSiamese = torch.load("./utils/runs/TriSiamese_TripletMarginLoss/run0/TriSiamese_TripletMarginLoss_train_loss.pt")
    acc_Triplet_TriSiamese = torch.load("./utils/runs/TriSiamese_TripletMarginLoss/run0/TriSiamese_TripletMarginLoss_train_acc.pt")

    loss_Contrastive_TriSiamese = torch.load(
        "./utils/runs/TriSiamese_ContrastiveLoss/run0/TriSiamese_ContrastiveLoss_train_loss.pt")
    acc_Contrastive_TriSiamese = torch.load(
        "./utils/runs/TriSiamese_ContrastiveLoss/run0/TriSiamese_ContrastiveLoss_train_acc.pt")

    loss_second_pretrain_BiSiamese = torch.load(
        "./utils/runs/BiSiamese_CrossEntropyLoss/run0/BiSiamese_CrossEntropyLoss_train_loss.pt")
    acc_second_pretrain_BiSiamese = torch.load(
        "./utils/runs/BiSiamese_CrossEntropyLoss/run0/BiSiamese_CrossEntropyLoss_train_acc.pt")

    loss_finally_BiSiamese = torch.load(
        "./utils/runs/BiSiamese_CrossEntropyLoss/run1/BiSiamese_CrossEntropyLoss_train_loss.pt")
    acc_finally_BiSiamese = torch.load("./utils/runs/BiSiamese_CrossEntropyLoss/run1/BiSiamese_CrossEntropyLoss_train_acc.pt")

    loss_pretrain_LeNet_dict = {i: loss_pretrain_LeNet[i] for i in range(len(loss_pretrain_LeNet))}
    acc_pretrain_LeNet_dict = {i: acc_pretrain_LeNet[i] for i in range(len(acc_pretrain_LeNet))}

    loss_Triplet_TriSiamese_dict = {i: loss_Triplet_TriSiamese[i] for i in range(len(loss_Triplet_TriSiamese))}
    acc_Triplet_TriSiamese_dict = {i: acc_Triplet_TriSiamese[i] for i in range(len(acc_Triplet_TriSiamese))}

    loss_Contrastive_TriSiamese_dict = {i: loss_Contrastive_TriSiamese[i] for i in
                                        range(len(loss_Contrastive_TriSiamese))}
    acc_Contrastive_TriSiamese_dict = {i: acc_Contrastive_TriSiamese[i] for i in range(len(acc_Contrastive_TriSiamese))}

    loss_second_pretrain_BiSiamese_dict = {i: loss_second_pretrain_BiSiamese[i] for i in
                                           range(len(loss_second_pretrain_BiSiamese))}
    acc_second_pretrain_BiSiamese_dict = {i: acc_second_pretrain_BiSiamese[i] for i in
                                          range(len(acc_second_pretrain_BiSiamese))}

    loss_finally_BiSiamese_dict = {i: loss_finally_BiSiamese[i] for i in range(len(loss_finally_BiSiamese))}
    acc_finally_BiSiamese_dict = {i: acc_finally_BiSiamese[i] for i in range(len(acc_finally_BiSiamese))}

    for step, los in loss_pretrain_LeNet_dict.items():
        writer.add_scalars('loss_pretrain_LeNet', {'loss_pretrain_LeNet': los}, global_step=step)
    for step, los in loss_Triplet_TriSiamese_dict.items():
        writer.add_scalars('loss_Triplet_TriSiamese', {'loss_Triplet_TriSiamese': los}, global_step=step)
    for step, los in loss_Contrastive_TriSiamese_dict.items():
        writer.add_scalars('loss_Contrastive_TriSiamese_dict', {'loss_Contrastive_TriSiamese_dict': los},
                           global_step=step)
    for step, los in loss_second_pretrain_BiSiamese_dict.items():
        writer.add_scalars('loss_second_pretrain_BiSiamese', {'loss_second_pretrain_BiSiamese': los}, global_step=step)
    for step, los in loss_finally_BiSiamese_dict.items():
        writer.add_scalars('loss_finally_BiSiamese', {'loss_finally_BiSiamese': los}, global_step=step)

    for step, acc in acc_pretrain_LeNet_dict.items():
        writer.add_scalars('acc_pretrain_LeNet', {'acc_pretrain_LeNet': acc}, global_step=step)
    for step, acc in acc_Triplet_TriSiamese_dict.items():
        writer.add_scalars('acc_Triplet_TriSiamese', {'acc_Triplet_TriSiamese': acc}, global_step=step)
    for step, acc in acc_Contrastive_TriSiamese_dict.items():
        writer.add_scalars('acc_Contrastive_TriSiamese_dict', {'acc_Contrastive_TriSiamese_dict': acc},
                           global_step=step)
    for step, acc in acc_second_pretrain_BiSiamese_dict.items():
        writer.add_scalars('acc_second_pretrain_BiSiamese', {'acc_second_pretrain_BiSiamese': acc}, global_step=step)
    for step, acc in acc_finally_BiSiamese_dict.items():
        writer.add_scalars('acc_finally_BiSiamese', {'acc_finally_BiSiamese': acc}, global_step=step)
