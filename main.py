#  用于调试无法使用jupyter notebook的环境
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2


# Network Architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.BatchNorm1d(120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.BatchNorm1d(84), nn.ReLU(), nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


print("whether to support cuda")
print(torch.cuda.is_available())  # 输出是否支持cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动识别使用cuda还是cpu
batch_size = 64
LR = 0.001
train_dataset = datasets.MNIST(root="E:\\pytorch_study\\lab\\data\\", train=True, transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root="E:\\pytorch_study\\lab\\data\\", train=False, transform=transforms.ToTensor(),
                              download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# 装载测试集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

# 建立一个数据迭代器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    net.parameters(),
    lr=LR,
)

print('准备开始测试')
epoch = 1
if __name__ == '__main__':
    for epoch in range(epoch):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0