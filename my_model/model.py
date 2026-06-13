import torch
import torch.nn as nn

# 定义简单的 Residual Block
class SimpleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation_str='relu'):
        super(SimpleResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 激活函数选择器
        if activation_str == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation_str == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)
            
        # 残差快捷连接 (Shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out

# 自定义 ResNet 网络模型
class ModularResNet(nn.Module):
    def __init__(self, filters=[64, 128, 256], activation_str='relu', num_classes=10, dropout_rate=0.3):
        super(ModularResNet, self).__init__()
        self.in_channels = filters[0]
        
        # 初始卷积层和BatchNorm层
        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.act = nn.ReLU(inplace=True) if activation_str == 'relu' else (nn.GELU() if activation_str == 'gelu' else nn.LeakyReLU(0.1))
        
        # 残差块层
        self.layer1 = self._make_layer(filters[0], stride=1, act=activation_str)
        self.layer2 = self._make_layer(filters[1], stride=2, act=activation_str)
        self.layer3 = self._make_layer(filters[2], stride=2, act=activation_str)
        
        # 池化层和全连接层
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(filters[2], num_classes)

    def _make_layer(self, out_channels, stride, act):
        layer = SimpleResBlock(self.in_channels, out_channels, stride, act)
        self.in_channels = out_channels
        return layer

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 返回模型可训练参数数量 (单位：百万)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

