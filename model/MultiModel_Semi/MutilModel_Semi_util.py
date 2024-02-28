import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import ce_loss
import numpy as np


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


# def consistency_loss(logits_w, class_acc, it, ds, p_cutoff, use_flex=False):
#     pseudo_label = torch.softmax(logits_w, dim=-1)
#     max_probs, max_idx = torch.max(pseudo_label, dim=-1)
#     if use_flex:
#         mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()
#     else:
#         mask = max_probs.ge(p_cutoff).float()
#     select = max_probs.ge(p_cutoff).long()

#     return (ce_loss(logits_w, max_idx.detach(), use_hard_labels=True,
#                     reduction='none') * mask).mean(), select, max_idx.long()


def cross_entropy_loss(logits, labels, use_hard_labels=True, reduction='mean'):
    if use_hard_labels:
        loss = F.cross_entropy(logits, labels, reduction=reduction)
    else:
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(log_probs * labels).sum(dim=-1)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
    return loss


def consistency_loss(logits_x_ulb_w, time_p, p_model, use_hard_labels=True):
    pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    p_cutoff = time_p
    p_model_cutoff = p_model / torch.max(p_model, dim=-1)[0]
    mask = max_probs.ge(p_cutoff * p_model_cutoff[max_idx])
    if use_hard_labels:
        masked_loss = cross_entropy_loss(logits_x_ulb_w, max_idx, use_hard_labels) * mask.float()
    else:
        pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
        masked_loss = cross_entropy_loss(logits_x_ulb_w, pseudo_label, use_hard_labels) * mask.float()
    return masked_loss.mean()




















class AdaptiveMultiModalConsistencyLoss(nn.Module):
    def __init__(self):
        super(AdaptiveMultiModalConsistencyLoss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.pairwise_loss = nn.MSELoss()

    def forward(self, modality1, modality2, modality3):
        loss12 = self.pairwise_loss(modality1, modality2)
        loss23 = self.pairwise_loss(modality2, modality3)
        loss31 = self.pairwise_loss(modality3, modality1)
        loss = self.alpha * loss12 + self.beta * loss23 + self.gamma * loss31

        labels1 = torch.argmax(modality1, dim=1)
        labels2 = torch.argmax(modality2, dim=1)
        labels3 = torch.argmax(modality3, dim=1)

        # 找出至少两个模态同意的类别
        pseudo_labels = torch.where(labels1 == labels2, labels1, labels2)
        pseudo_labels = torch.where(pseudo_labels == labels3, pseudo_labels, torch.full_like(pseudo_labels, 0))
        return loss, pseudo_labels

class MultiModalSupervisedLoss(nn.Module):
    def __init__(self):
        super(MultiModalSupervisedLoss, self).__init__()
        self.delta1 = nn.Parameter(torch.tensor(1.0))
        self.delta2 = nn.Parameter(torch.tensor(1.0))
        self.delta3 = nn.Parameter(torch.tensor(1.0))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data1, data2, data3, labels):
        loss1 = self.loss_fn(data1, labels)
        loss2 = self.loss_fn(data2, labels)
        loss3 = self.loss_fn(data3, labels)

        # 加权求和
        total_loss = self.delta1 * loss1 + self.delta2 * loss2 + self.delta3 * loss3

        return total_loss
    

def Supervised_label(data1, data2, data3): # data1 为总预测概率
    device = data1.device
    pred_x1 = torch.softmax(data1,dim=-1)
    pred_x2 = torch.softmax(data2,dim=-1)
    pred_x3 = torch.softmax(data3,dim=-1)

    batch_size = data1.size(0)
    num_classes = data1.size(1)
    votes = torch.zeros(batch_size, num_classes, dtype=torch.int64,device=device)
    for idx in range(batch_size):
        votes[idx, torch.argmax(pred_x1[idx])] += 1
        votes[idx, torch.argmax(pred_x2[idx])] += 1
        votes[idx, torch.argmax(pred_x3[idx])] += 1

    final_labels = torch.argmax(votes, dim=1)
    # 计算预测概率
    avg_probs = (pred_x1 + pred_x2 + pred_x3) / 3

    # print("Final labels:", final_labels)
    # print("Max label:", torch.max(final_labels))
    # print("Min label:", torch.min(final_labels))

    final_probs = torch.gather(avg_probs, 1, final_labels.unsqueeze(1)).squeeze(1)
    
    return final_labels, final_probs

