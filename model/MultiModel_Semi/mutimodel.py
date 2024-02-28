import torch
import torch.nn as nn
from model.nets.resnet import resnet18
from model.nets.wrn import WideResNet
from model.nets.mobilenetv2 import MobileNetV2


class MultiModalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiModalModel, self).__init__()

        # 模态 1
        # self.modal1 = resnet18(num_classes)
        # self.modal1.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))
        ## self.modal1.fc = nn.Identity()
        self.modal1 = WideResNet()
        self.modal1.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1,
                               padding=1, bias=True)
        
        # self.modal1.fc = nn.Identity()

        # 模态 2
        # self.modal2 = resnet18(num_classes)
        # self.modal2.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))
        # self.modal2.fc = nn.Identity()
        self.modal2 = WideResNet()
        self.modal2.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1,
                               padding=1, bias=True)
        
        # self.modal2.fc = nn.Identity()

        # # 模态 3
        self.modal3 = WideResNet()
        self.modal3.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1,
                               padding=1, bias=True)
        # self.modal3.fc = nn.Identity()

        # 跨模态融合
        # self.fc_fusion = nn.Conv2d(1280*2, num_classes, 1)
        self.fc_fusion = nn.Linear(3*2, num_classes)

    def forward(self, x, return_modal_outputs=False):
        x1 = self.modal1(x[:,0,:,:].reshape(-1,1,32,32))
        x2 = self.modal2(x[:,1,:,:].reshape(-1,1,32,32))
        x3 = self.modal3(x[:,2,:,:].reshape(-1,1,32,32))
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.subplot(1,3,1)
        # plt.imshow(x[0,0,:,:].cpu())
        # plt.subplot(1,3,2)
        # plt.imshow(x[0,1,:,:].cpu())
        # plt.subplot(1,3,3)
        # plt.imshow(np.transpose(x[0].cpu(),(1, 2, 0)))
        # plt.savefig('./x.png')


        # x1 = self.modal1(x[:,0,:16,:].reshape(-1,1,16,32))
        # x2 = self.modal2(x[:,0,16:,:].reshape(-1,1,16,32))


        # 特征级融合
        x = torch.cat((x1, x2, x3), dim=1)
        # x = torch.cat((x1, x2), dim=1)


        # 全连接层
        x = self.fc_fusion(x)
        x = x.view(x.size(0), -1)

        # if return_modal_outputs:
            

            # return x, x1, x2, x3
            # return x, x1_output, x2_output
        # else:
            # return x
        return x


class build_multimodel18:
    def __init__(self, is_remix=False):
        self.is_remix = is_remix

    def build(self, num_classes):
        return MultiModalModel(num_classes=num_classes)