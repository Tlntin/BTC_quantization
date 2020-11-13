import torch
from torch import nn, optim
import torch.nn.functional as F
from config import Config


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )
        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))

        # Excitation
        out = out * w
        out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, num_classes=Config.classify_num):
        """
        基于Resnet升级版SeNet构建的网络
        input_size = (batch_size, 256, 32)
        :param block: 残差网络块
        :param num_classes: 分类结果,目前为5分类
        """
        super().__init__()
        self.rnn = nn.LSTM(32, 64, num_layers=2, dropout=0.4, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.in_planes = 32
        self.layer0 = self._make_layer(block, 32, 2, stride=1)
        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)  # [28, 254, 128]
        out = out.view(out.size(0), 1, out.size(1), out.size(2))  # 增加一维
        out = F.relu(self.bn1(self.conv1(out)))  # [28, 32, 64, 64]
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size(2))  # [28, 512, 1, 1]
        out = out.view(out.size(0), -1)  # [28, 512]
        out = self.linear(out)
        return out


def resnet18():
    return SENet(PreActBlock)


if __name__ == '__main__':
    x1 = torch.rand(28, 256, 32)
    net = SENet(PreActBlock)
    result = net(x1)
    print(result.shape)
