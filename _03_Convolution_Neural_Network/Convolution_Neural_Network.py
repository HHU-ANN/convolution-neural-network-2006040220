# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
    

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=128,kernel_size=(3,3),stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=(3,3),stride=2),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(5,5),stride=1,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=(3,3),stride=2),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=384,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3),stride=2)
                                    )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*3*3, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )
 
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    pass

def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(root='data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

def main():
    model = NeuralNetwork() # 若有参数则传入参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    return model
    
