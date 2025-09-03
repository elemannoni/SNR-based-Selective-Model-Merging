import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SeparableConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_pointwise = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = self.pointwise(x)
        x = self.bn_pointwise(x)
        return x

class ResidualSeparableBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualSeparableBlock, self).__init__()
        self.conv1 = SeparableConvBlock(channels, channels)
        self.conv2 = SeparableConvBlock(channels, channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return F.relu(out)


class CNN_snr(nn.Module):
    def __init__(self, output_size=10):
        super(CNN_snr, self).__init__()
        self.initial_conv = SeparableConvBlock(3, 64)
        self.res_stage1 = nn.Sequential(
            ResidualSeparableBlock(channels=64),
            ResidualSeparableBlock(channels=64),
            ResidualSeparableBlock(channels=64)
        )
        self.transition1 = SeparableConvBlock(64, 128)
        self.res_stage2 = nn.Sequential(
            ResidualSeparableBlock(channels=128),
            ResidualSeparableBlock(channels=128),
            ResidualSeparableBlock(channels=128),
            ResidualSeparableBlock(channels=128)
        )
        self.transition2 = SeparableConvBlock(128, 256)
        self.res_stage3 = nn.Sequential(
            ResidualSeparableBlock(channels=256),
            ResidualSeparableBlock(channels=256),
            ResidualSeparableBlock(channels=256)
        )
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn_fc4 = nn.BatchNorm1d(64)
        self.output_layer = nn.Linear(64, output_size)

        layers_list = []
        layers_list.extend(self._expand_sep_block(self.initial_conv))
        for block in self.res_stage1:
            layers_list.extend(self._expand_res_block(block))
        layers_list.extend(self._expand_sep_block(self.transition1))
        for block in self.res_stage2:
            layers_list.extend(self._expand_res_block(block))
        layers_list.extend(self._expand_sep_block(self.transition2))
        for block in self.res_stage3:
            layers_list.extend(self._expand_res_block(block))
        layers_list.extend([
            self.fc1, self.bn_fc1, self.fc2, self.bn_fc2,
            self.fc3, self.bn_fc3, self.fc4, self.bn_fc4
        ])
        self.layers = nn.ModuleList(layers_list)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)

    def _expand_sep_block(self, block):
        """Helper per estrarre i layer da un SeparableConvBlock."""
        return [
            block.depthwise, block.bn_depthwise,
            block.pointwise, block.bn_pointwise
        ]

    def _expand_res_block(self, block):
        """Helper per estrarre i layer da un ResidualSeparableBlock."""
        return [
            *self._expand_sep_block(block.conv1),
            *self._expand_sep_block(block.conv2)
        ]

    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        x = self.res_stage1(x)
        x = self.pool(x)
        x = self.transition1(x)
        x = self.res_stage2(x)
        x = self.pool(x)
        x = self.transition2(x)
        x = self.res_stage3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn_fc3(self.fc3(x))))
        x = F.relu(self.bn_fc4(self.fc4(x)))
        return self.output_layer(x)
