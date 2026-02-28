from functools import partial

import torch
import torch.nn as nn
import time
# 定义是否使用CUDA
use_cuda = torch.cuda.is_available()  # 自动检测是否有GPU
device = torch.device('cuda' if use_cuda else 'cpu')

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
    
    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

class DepthwiseConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        return self.depthwise(x)

class PointwiseConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, **kwargs),
            nn.BatchNorm2d(output_channels)
        )
    
    def forward(self, x):
        return self.pointwise(x)

class ShuffleNetUnit(nn.Module):

    def __init__(self, input_channels, output_channels, stage, stride, groups):
        super().__init__()

        self.gp1 = nn.Sequential(
            PointwiseConv2d(
                input_channels, 
                int(output_channels / 4), 
                groups=groups
            ),
            nn.ReLU(inplace=True)
        )

        
        self.channel_shuffle = ChannelShuffle(groups)

        self.depthwise = DepthwiseConv2d(
            int(output_channels / 4), 
            int(output_channels / 4), 
            3, 
            groups=int(output_channels / 4), 
            stride=stride,
            padding=1
        )

        self.expand = PointwiseConv2d(
            int(output_channels / 4),
            output_channels,
            groups=groups
        )

        self.relu = nn.ReLU(inplace=True)
        self.fusion = self._add
        self.shortcut = nn.Sequential()

        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)

            self.expand = PointwiseConv2d(
                int(output_channels / 4),
                output_channels - input_channels,
                groups=groups
            )

            self.fusion = self._cat
    
    def _add(self, x, y):
        return torch.add(x, y)
    
    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        shortcut = self.shortcut(x)

        shuffled = self.gp1(x)
        shuffled = self.channel_shuffle(shuffled)
        shuffled = self.depthwise(shuffled)
        shuffled = self.expand(shuffled)

        output = self.fusion(shortcut, shuffled)
        output = self.relu(output)

        return output

class ShuffleNet(nn.Module):

    def __init__(self, num_blocks, num_classes=5, groups=3):
        super().__init__()

        if groups == 1:
            out_channels = [24, 144, 288, 567]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]

        self.conv1 = BasicConv2d(3, out_channels[0], 3, padding=1, stride=1)
        self.input_channels = out_channels[0]

        self.stage2 = self._make_stage(
            ShuffleNetUnit, 
            num_blocks[0], 
            out_channels[1], 
            stride=2, 
            stage=2,
            groups=groups
        )

        self.stage3 = self._make_stage(
            ShuffleNetUnit, 
            num_blocks[1], 
            out_channels[2], 
            stride=2,
            stage=3, 
            groups=groups
        )

        self.stage4 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[2],
            out_channels[3],
            stride=2,
            stage=4,
            groups=groups
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_stage(self, block, num_blocks, output_channels, stride, stage, groups):

        strides = [stride] + [1] * (num_blocks - 1)

        stage = []

        for stride in strides:
            stage.append(
                block(
                    self.input_channels, 
                    output_channels, 
                    stride=stride, 
                    stage=stage, 
                    groups=groups
                )
            )
            self.input_channels = output_channels

        return nn.Sequential(*stage)

def shufflenet():
    return ShuffleNet([4, 8, 4])


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
    model = shufflenet()

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