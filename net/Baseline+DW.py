import torch
import torch.nn as nn
from torchvision.models import ResNet


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
    # def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):

        super(SEBasicBlock, self).__init__()

        self.conv1 = DWConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv2 = DWConv(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)

        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def Baseline(num_classes=5):
    model = ResNet(SEBasicBlock, [5, 5, 5, 5], num_classes=num_classes) # 每个 SEBasicBlock 有 4 个卷积层，总共有 10 个 SEBasicBlock
    return model


if __name__ == "__main__":
    from torchinfo import summary
    model = Baseline(num_classes=5)

    # 打印网络结构图
    summary(model, input_size=(1, 3, 200, 200), device="cpu",
            col_names=["input_size", "output_size", "num_params", 'mult_adds'])

    # 计算参数
    from thop import profile
    input = torch.randn(1, 3, 200, 200)
    flops, parms = profile(model, inputs=(input, ))
    print(f"FLOPs:{flops/1e9}G,params:{parms/1e6}M")

    img = torch.randn(1, 3, 200, 200)
    out = model(img)
    print(out.shape)
