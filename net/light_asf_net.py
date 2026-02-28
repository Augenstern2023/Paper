import torch
import torch.nn as nn
import torch.nn.functional as F

# --- EfficientNet风格的SE模块 ---


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_factor=4):
        super().__init__()
        squeeze_channels = in_channels // squeeze_factor
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.act1(scale)
        scale = self.fc2(scale)
        scale = self.act2(scale)
        return x * scale

# --- 轻量化的MBConv Block ---


class LightMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2, kernel_size=3, stride=1, se_ratio=0.25, drop_rate=0.):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        # 减少内存消耗：只在需要时扩展
        if expand_ratio != 1:
            self.expand = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
            self.bn0 = nn.BatchNorm2d(mid_channels)
            self.act0 = nn.SiLU()
        else:
            self.expand = nn.Identity()
            self.bn0 = nn.Identity()
            self.act0 = nn.Identity()

        self.dwconv = nn.Conv2d(mid_channels, mid_channels, kernel_size,
                                stride, kernel_size//2, groups=mid_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.SiLU()

        # 轻量化SE模块
        self.se = SqueezeExcitation(
            mid_channels) if se_ratio > 0 else nn.Identity()

        self.project = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop_rate = drop_rate

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.bn0(out)
        out = self.act0(out)
        out = self.dwconv(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.se(out)
        out = self.project(out)
        out = self.bn2(out)
        if self.use_res_connect:
            if self.drop_rate > 0 and self.training:
                out = F.dropout(out, p=self.drop_rate, training=True)
            out = out + identity
        return out

# --- AFF融合模块 ---


class AFFModule(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc_weight = nn.Linear(in_features * 2, 2)
        nn.init.constant_(self.fc_weight.weight, 0)
        nn.init.constant_(self.fc_weight.bias, 0)

    def forward(self, local_feature, global_feature):
        concat = torch.cat([local_feature, global_feature], dim=1)
        weights = self.fc_weight(concat)
        weights = F.softmax(weights, dim=1)
        local_weight = weights[:, 0:1]
        global_weight = weights[:, 1:2]
        fused = local_feature * local_weight + global_feature * global_weight
        return fused

# --- 轻量化主干网络 ---


class LightASF_former_S(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=4, embed_dim=256, depth=3, num_heads=8, mlp_ratio=3., qkv_bias=False, drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        # 轻量化Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 16, kernel_size=3, stride=2,
                      padding=1, bias=False),  # 减少初始通道数
            nn.BatchNorm2d(16),
            nn.SiLU()
        )

        # 分支1：ASF-former（轻量化）
        from models.asf_former import ASF_former
        self.asf_branch = ASF_former(
            img_size=img_size//2,  # 因为stem下采样
            tokens_type='performer',
            in_chans=16,  # 减少输入通道
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            token_dim=32,  # 减少token维度
            ASF=True
        )

        # 分支2：轻量化MBConv堆叠
        mbconv_blocks = []
        mbconv_in = 16  # 从16开始
        for i in range(depth):
            # 逐步增加通道数，避免内存爆炸
            if i == 0:
                mbconv_out = 64
            elif i == 1:
                mbconv_out = 128
            else:
                mbconv_out = embed_dim

            mbconv_blocks.append(LightMBConvBlock(
                in_channels=mbconv_in,
                out_channels=mbconv_out,
                expand_ratio=2,  # 减少扩展比
                kernel_size=3,
                stride=1,
                se_ratio=0.25,
                drop_rate=drop_rate
            ))
            mbconv_in = mbconv_out

        self.mbconv_blocks = nn.Sequential(*mbconv_blocks)
        self.mbconv_pool = nn.AdaptiveAvgPool2d(1)
        self.mbconv_fc = nn.Linear(embed_dim, num_classes)

        # 融合
        self.aff_fusion = AFFModule(num_classes)

    def forward(self, x):
        x = self.stem(x)

        # 分支1
        asf_out = self.asf_branch(x)  # [B, num_classes]

        # 分支2
        mbconv_feat = self.mbconv_blocks(x)
        mbconv_feat = self.mbconv_pool(mbconv_feat).flatten(1)
        mbconv_out = self.mbconv_fc(mbconv_feat)

        # 融合
        out = self.aff_fusion(mbconv_out, asf_out)
        return out


# 测试代码
if __name__ == '__main__':
    import time
    import torch

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模型
    model = LightASF_former_S(
        img_size=224, in_chans=3, num_classes=4, embed_dim=256, depth=3)
    model = model.to(device)
    model.eval()

    # 测试输入
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224).to(device)

    # 1. 测试输出形状
    print("=" * 50)
    print("1. 输出形状测试")
    with torch.no_grad():
        out = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {out.shape}')
    assert out.shape == (batch_size, model.asf_branch.num_classes)
    print('✓ 输出形状测试通过!')

    # 2. 参数量统计
    print("\n" + "=" * 50)
    print("2. 参数量统计")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {non_trainable_params:,}")

    # 按模块统计参数
    print("\n各模块参数量:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name}: {params:,}")

    # 3. 推理速度测试
    print("\n" + "=" * 50)
    print("3. 推理速度测试")

    # 预热
    print("预热中...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)

    # 测试推理速度
    num_runs = 100
    times = []

    print(f"进行 {num_runs} 次推理测试...")
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()  # 确保GPU计算完成
            end_time = time.time()
            times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"平均推理时间: {avg_time*1000:.2f} ms")
    print(f"最快推理时间: {min_time*1000:.2f} ms")
    print(f"最慢推理时间: {max_time*1000:.2f} ms")
    print(f"FPS: {1/avg_time:.2f}")

    # 4. 内存使用量统计
    print("\n" + "=" * 50)
    print("4. 内存使用量统计")

    if device.type == 'cuda':
        # GPU内存统计
        torch.cuda.empty_cache()  # 清理缓存
        initial_memory = torch.cuda.memory_allocated()

        with torch.no_grad():
            _ = model(x)

        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()

        print(f"初始GPU内存: {initial_memory / 1024**2:.2f} MB")
        print(f"峰值GPU内存: {peak_memory / 1024**2:.2f} MB")
        print(f"当前GPU内存: {current_memory / 1024**2:.2f} MB")
        print(f"推理占用内存: {(peak_memory - initial_memory) / 1024**2:.2f} MB")

        # 总GPU内存
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU总内存: {total_gpu_memory / 1024**3:.2f} GB")
        print(f"内存使用率: {peak_memory / total_gpu_memory * 100:.2f}%")
    else:
        print("CPU模式，跳过内存统计")

    # 5. 模型大小计算
    print("\n" + "=" * 50)
    print("5. 模型大小计算")

    # 计算模型文件大小（假设float32）
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per parameter
    print(f"模型文件大小 (float32): {model_size_mb:.2f} MB")

    # 如果使用float16
    model_size_mb_fp16 = total_params * 2 / \
        (1024 * 1024)  # 2 bytes per parameter
    print(f"模型文件大小 (float16): {model_size_mb_fp16:.2f} MB")

    # 6. 性能总结
    print("\n" + "=" * 50)
    print("6. 性能总结")
    print(f"✓ 模型: LightASF_former_S")
    print(f"✓ 输入尺寸: {x.shape[2]}x{x.shape[3]}")
    print(f"✓ 输出类别: {model.asf_branch.num_classes}")
    print(f"✓ 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
