import torch
import torch.nn as nn

import time
# 定义是否使用CUDA
use_cuda = torch.cuda.is_available()  # 自动检测是否有GPU
device = torch.device('cuda' if use_cuda else 'cpu')

class dw_conv(nn.Module):
    def __init__(self, in_dim, out_dim, stride):
        super(dw_conv, self).__init__()
        self.dw_conv_k3 = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, stride=stride, groups=in_dim, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw_conv_k3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class point_conv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(point_conv, self).__init__()
        self.p_conv_k1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.p_conv_k1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileNets(nn.Module):

    def __init__(self, num_classes, large_img):
        super(MobileNets, self).__init__()
        self.num_classes = num_classes
        if large_img:
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                dw_conv(32, 32, 1),
                point_conv(32, 64),
                dw_conv(64, 64, 2),
                point_conv(64, 128),
                dw_conv(128, 128, 1),
                point_conv(128, 128),
                dw_conv(128, 128, 2),
                point_conv(128, 256),
                dw_conv(256, 256, 1),
                point_conv(256, 256),
                dw_conv(256, 256, 2),
                point_conv(256, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 2),
                point_conv(512, 1024),
                dw_conv(1024, 1024, 2),
                point_conv(1024, 1024),
                nn.AvgPool2d(7),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                dw_conv(32, 32, 1),
                point_conv(32, 64),
                dw_conv(64, 64, 1),
                point_conv(64, 128),
                dw_conv(128, 128, 1),
                point_conv(128, 128),
                dw_conv(128, 128, 1),
                point_conv(128, 256),
                dw_conv(256, 256, 1),
                point_conv(256, 256),
                dw_conv(256, 256, 1),
                point_conv(256, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 512),
                dw_conv(512, 512, 1),
                point_conv(512, 1024),
                dw_conv(1024, 1024, 1),
                point_conv(1024, 1024),
                nn.AvgPool2d(4),
            )

        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def mobilenet(num_classes, large_img, **kwargs):
    r"""PyTorch implementation of the MobileNets architecture
    <https://arxiv.org/abs/1704.04861>`_.
    Model has been designed to work on either ImageNet or CIFAR-10
    Args:
        num_classes (int): 1000 for ImageNet, 10 for CIFAR-10
        large_img (bool): True for ImageNet, False for CIFAR-10
    """
    model = MobileNets(num_classes, large_img, **kwargs)
    if use_cuda:
        model = model.cuda()
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
    model = MobileNets(5, large_img=False)

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