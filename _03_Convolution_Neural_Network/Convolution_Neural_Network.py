import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
 
transform=torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_dataset=torchvision.datasets.CIFAR10('data/',train=True,transform=transform,download=True)
test_dataset=torchvision.datasets.CIFAR10('data/',train=False,transform=transform,download=True)
 
# length 长度
print('训练数据集长度: {}'.format(len(train_dataset)))
print('测试数据集长度: {}'.format(len(test_dataset)))
# DataLoader创建数据集
train_dataloader=DataLoader(train_dataset,batch_size=84,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=84,shuffle=True)
 
examples=enumerate(test_dataloader)#组合成一个索引序列
batch_idx,(example_data,example_targets)=next(examples)
classes=('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
fig=plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    #plt.tight_layout()
    img= example_data[i]
    print(img.shape)
    img = img.swapaxes(0, 1)
    img = img.swapaxes(1, 2)
    #img = img[:,:,::-1]
    plt.imshow(img,interpolation='none')
    plt.title('target: {}'.format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=128,kernel_size=(3,3),stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=(3,3),stride=2),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(5,5),stride=1,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=(3,3),stride=2),
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3),stride=2)
                                    )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*3*3, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )
 
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
 
net = NeuralNetwork()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
print("参数数：{}".format(sum(x.numel() for x in net.parameters())))
 
# 损失函数与优化器
loss=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.05)
 
# 记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
 
# 添加tensorboard
writer=SummaryWriter("./logs_train")
 
for epoch in range(25):
 
    print("——————第 {} 轮训练开始——————".format(epoch+1))
 
    #训练开始
    net.train()
 
    for imgs,targets in train_dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        output=net(imgs)
 
        Loss=loss(output,targets)
        # 优化器优化模型
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
 
        _, pred = output.max(1)
        num_correct = (pred == targets).sum().item()
        acc = num_correct / (64)
        total_train_step = total_train_step + 1
        if total_train_step%100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step,Loss.item()))
            writer.add_scalar("train_loss", Loss.item(), total_train_step)
            writer.add_scalar("train_acc", acc, total_train_step)
 
 
 
    # 测试步骤开始
    net.eval()
    eval_loss = 0
    eval_losses = 0
    eval_acc = 0
    eval_acces = 0
    with torch.no_grad():
        for imgs,targets in test_dataloader:
            imgs=imgs.to(device)
            targets=targets.to(device)
            output=net(imgs)
            Loss=loss(output,targets)
            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            eval_loss += Loss
            acc = num_correct / imgs.shape[0]
            eval_acc += acc
 
        eval_losses = eval_loss/(len(test_dataloader))
        eval_acces = eval_acc/(len(test_dataloader))
        print("整体测试集上的Loss: {}".format(eval_losses))
        print("整体测试集上的正确率: {}".format(eval_acces))
        writer.add_scalar("test_loss", eval_losses, total_test_step)
        writer.add_scalar("test_accuracy", eval_acces, total_test_step)
        total_test_step = total_test_step + 1
 
        torch.save(net, "tudui_{}.pth".format(epoch))
        print("模型已保存")
 
writer.close()
 
 