import torch
import torch.nn as nn
from torchvision.models import ResNet
import time

# 定义是否使用CUDA
use_cuda = torch.cuda.is_available()  # 自动检测是否有GPU
device = torch.device('cuda' if use_cuda else 'cpu')


class BatchNormm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5):
        super(BatchNormm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.bn = nn.BatchNorm2d(num_channels, eps=epsilon)

    def forward(self, x):
        return self.bn(x)

# class LayerNormm2D(nn.Module):
#     def __init__(self, num_channels, epsilon=1e-5):
#         super(LayerNormm2D, self).__init__()
#         self.num_channels = num_channels
#         self.epsilon = epsilon
#         self.ln = nn.LayerNorm(num_channels, eps=epsilon)
#
#     def forward(self, x):
#         # LayerNorm expects the shape [N, C, H, W]
#         # Transpose to [N, H, W, C] to apply LayerNorm, then back
#         x = x.permute(0, 2, 3, 1)
#         x = self.ln(x)
#         x = x.permute(0, 3, 1, 2)
#         return x


class LayerNormm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5):
        super(LayerNormm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        assert list(x.shape)[1] == self.num_channels
        assert len(x.shape) == 4  # 4 because len((batchsize, numchannels, height, width)) = 4
        variance, mean = torch.var(x, dim=[1, 2, 3], unbiased=False), torch.mean(x, dim=[1, 2, 3])

        out = (x - mean.view([-1, 1, 1, 1])) / torch.sqrt(variance.view([-1, 1, 1, 1]) + self.epsilon)
        return out


class BatchChannelNorm(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9):
        super(BatchChannelNorm, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.Batchh = BatchNormm2D(self.num_channels, epsilon=self.epsilon)
        self.layeer = LayerNormm2D(self.num_channels, epsilon=self.epsilon)
        # The BCN variable to be learnt
        self.BCN_var = nn.Parameter(torch.ones(self.num_channels))
        # Gamma and Beta for rescaling
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        X = self.Batchh(x)
        Y = self.layeer(x)
        out = self.BCN_var.view([1, self.num_channels, 1, 1]) * X + (
                1 - self.BCN_var.view([1, self.num_channels, 1, 1])) * Y
        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out


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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


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
        super(SEBasicBlock, self).__init__()

        self.conv1 = DWConv(inplanes, planes, stride=stride)
        self.bcn1 = BatchChannelNorm(planes)  # 使用 BatchChannelNorm 替换原来的 BatchNorm2d
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv2 = DWConv(planes, planes)
        self.bcn2 = BatchChannelNorm(planes)  # 使用 BatchChannelNorm 替换原来的 BatchNorm2d
        self.relu = nn.LeakyReLU(inplace=True)

        self.se = SELayer(planes, reduction)
        self.sa = SpatialAttention()
        self.bcn_se = BatchChannelNorm(planes)
        self.bcn_sa = BatchChannelNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bcn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bcn2(out)
        out = self.relu(out)

        out = self.se(out)
        out = self.bcn_se(out)  # 对SE后的输出进行归一化

        sa_out = self.sa(out)
        out = out * sa_out  # 将空间注意力作为加权因子应用于特征图
        out = self.bcn_sa(out)  # 对SA后的输出进行归一化

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def CassavaNet(num_classes=5):
    model = ResNet(SEBasicBlock, [4, 4, 4, 4], num_classes=num_classes)
    return model


def compute_fps(model, input_tensor, iterations=100):
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_iteration = total_time / iterations
    fps = 1.0 / avg_time_per_iteration

    return fps


if __name__ == "__main__":
    from torchinfo import summary
    model = CassavaNet(num_classes=5)

    # 打印网络结构图
    summary(model, input_size=(1, 3, 224, 224), device="cpu",
            col_names=["input_size", "output_size", "num_params", 'mult_adds'])

    # 计算参数
    from thop import profile
    input = torch.randn(1, 3, 224, 224)
    flops, parms = profile(model, inputs=(input, ))
    print(f"FLOPs:{flops/1e9}G,params:{parms/1e6}M")

    img = torch.randn(1, 3, 224, 224)
    out = model(img)
    print(out.shape)

    # 计算FPS
    fps = compute_fps(model, input)
    print(f"cuda:{use_cuda}==>FPS: {fps:.4f}")

