# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.layers import trunc_normal_
# from models.asf_former import ASF_former_S


# class ASF_Block(nn.Module):
#     def __init__(self, dim, num_heads=6, mlp_ratio=3., qkv_bias=False, drop=0., attn_drop=0.):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(
#             dim, num_heads, dropout=attn_drop, bias=qkv_bias)
#         self.norm2 = nn.LayerNorm(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Dropout(drop),
#             nn.Linear(mlp_hidden_dim, dim),
#             nn.Dropout(drop)
#         )

#     def forward(self, x):
#         # 自注意力
#         x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
#         # MLP
#         x = x + self.mlp(self.norm2(x))
#         return x


# class T2T_Module(nn.Module):
#     def __init__(self, img_size=224, in_chans=3, embed_dim=384, token_dim=64):
#         super().__init__()
#         self.soft_split1 = nn.Unfold(kernel_size=(
#             7, 7), stride=(4, 4), padding=(2, 2))
#         self.soft_split2 = nn.Unfold(kernel_size=(
#             3, 3), stride=(2, 2), padding=(1, 1))

#         # 简化的特征提取
#         self.proj1 = nn.Sequential(
#             nn.Linear(in_chans * 7 * 7, token_dim),
#             nn.LayerNorm(token_dim),
#             nn.GELU()
#         )
#         self.proj2 = nn.Sequential(
#             nn.Linear(token_dim * 3 * 3, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.GELU()
#         )

#         self.num_patches = (img_size // (4 * 2)) * (img_size // (4 * 2))

#     def forward(self, x):
#         # 第一次特征提取
#         x = self.soft_split1(x).transpose(1, 2)
#         x = self.proj1(x)

#         # 重塑并第二次特征提取
#         B, HW, C = x.shape
#         H = W = int((HW) ** 0.5)
#         x = x.transpose(1, 2).reshape(B, C, H, W)
#         x = self.soft_split2(x).transpose(1, 2)
#         x = self.proj2(x)

#         return x


# class BatchChannelNorm(nn.Module):
#     def __init__(self, num_channels, epsilon=1e-5, momentum=0.9):
#         super(BatchChannelNorm, self).__init__()
#         self.num_channels = num_channels
#         self.epsilon = epsilon
#         self.momentum = momentum
#         self.bn = nn.BatchNorm2d(num_channels, eps=epsilon)
#         self.ln = nn.LayerNorm(num_channels, eps=epsilon)
#         self.BCN_var = nn.Parameter(torch.ones(num_channels))
#         self.gamma = nn.Parameter(torch.ones(num_channels))
#         self.beta = nn.Parameter(torch.zeros(num_channels))

#     def forward(self, x):
#         bn_out = self.bn(x)
#         ln_out = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         out = self.BCN_var.view(1, -1, 1, 1) * bn_out + \
#             (1 - self.BCN_var.view(1, -1, 1, 1)) * ln_out
#         out = self.gamma.view(1, -1, 1, 1) * out + self.beta.view(1, -1, 1, 1)
#         return out


# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv1 = nn.Conv2d(
#             2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return x * self.sigmoid(x)


# class DWConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super(DWConv, self).__init__()
#         self.dw_conv = nn.Conv2d(
#             in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
#         self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
#         self.bcn = BatchChannelNorm(out_channels)
#         self.act = nn.LeakyReLU(inplace=True)

#     def forward(self, x):
#         x = self.dw_conv(x)
#         x = self.pw_conv(x)
#         x = self.bcn(x)
#         x = self.act(x)
#         return x


# class EnhancedTokenBlock(nn.Module):
#     def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(
#             dim, num_heads, dropout=attn_drop, bias=qkv_bias)

#         # 添加空间注意力
#         self.spatial_attn = SpatialAttention()

#         # 添加通道注意力
#         self.channel_attn = SELayer(dim)

#         # 使用深度可分离卷积进行特征提取
#         self.conv = DWConv(dim, dim)

#         self.norm2 = nn.LayerNorm(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout(drop),
#             nn.Linear(mlp_hidden_dim, dim),
#             nn.Dropout(drop)
#         )

#     def forward(self, x):
#         # 自注意力
#         norm_x = self.norm1(x)
#         attn_output, _ = self.attn(norm_x, norm_x, norm_x)
#         x = x + attn_output

#         # 调整维度以适应空间注意力
#         B, L, C = x.shape
#         H = W = int((L) ** 0.5)
#         x_2d = x.transpose(1, 2).reshape(B, C, H, W)

#         # 空间注意力
#         spatial_attn = self.spatial_attn(x_2d)
#         x_2d = x_2d * spatial_attn

#         # 通道注意力
#         x_2d = self.channel_attn(x_2d)

#         # 深度可分离卷积
#         x_2d = x_2d + self.conv(x_2d)

#         # 恢复原始维度
#         x = x_2d.reshape(B, C, L).transpose(1, 2)

#         # MLP
#         x = x + self.mlp(self.norm2(x))

#         return x


# class LightASF_former_S(nn.Module):
#     def __init__(self, num_classes=5, img_size=224, in_chans=3, depth=7, embed_dim=384):
#         super().__init__()

#         # 使用原始ASF-former的token生成部分
#         self.base_model = ASF_former_S(
#             num_classes=num_classes,
#             img_size=img_size,
#             in_chans=in_chans
#         )

#         # 替换原始transformer blocks为增强版本
#         self.blocks = nn.ModuleList([
#             EnhancedTokenBlock(
#                 dim=embed_dim,
#                 num_heads=6,
#                 mlp_ratio=3.,
#                 qkv_bias=True,
#                 drop=0.1,
#                 attn_drop=0.1
#             ) for _ in range(depth)
#         ])

#         # 添加特征增强层
#         self.feature_enhancement = nn.Sequential(
#             DWConv(embed_dim, embed_dim),
#             SELayer(embed_dim)
#         )
#         self.spatial_attention = SpatialAttention()

#         # 修改分类头结构
#         self.norm = nn.LayerNorm(embed_dim)
#         self.fc = nn.Linear(embed_dim, num_classes)

#     def forward_features(self, x):
#         # 获取基础特征
#         tokens, w_attn_depth, w_conv_depth = self.base_model.tokens_to_token(x)
#         # print(f"After tokens_to_token: {tokens.shape}")

#         # 应用增强的transformer blocks
#         for block in self.blocks:
#             tokens = block(tokens)
#         # print(f"After transformer blocks: {tokens.shape}")

#         # 调整维度以适应特征增强层
#         B, L, C = tokens.shape
#         H = W = int((L) ** 0.5)
#         # print(f"B: {B}, L: {L}, C: {C}, H: {H}, W: {W}")

#         # 确保维度匹配
#         if H * W != L:
#             # 如果维度不匹配，使用线性插值调整大小
#             tokens = tokens.transpose(1, 2)  # [B, C, L]
#             tokens = tokens.reshape(B, C, H, W)
#         else:
#             tokens = tokens.transpose(1, 2).reshape(B, C, H, W)
#         # print(f"After reshape: {tokens.shape}")

#         # 特征增强
#         tokens = self.feature_enhancement(tokens)
#         spatial_attn = self.spatial_attention(tokens)
#         tokens = tokens * spatial_attn
#         # print(f"After feature enhancement: {tokens.shape}")

#         # 全局平均池化
#         tokens = tokens.mean(dim=[2, 3])  # [B, C]
#         # print(f"After global pooling: {tokens.shape}")

#         return tokens

#     def forward(self, x):
#         x = self.forward_features(x)  # [B, C]
#         # print(f"Before norm: {x.shape}")
#         x = self.norm(x)  # [B, C]
#         # print(f"After norm: {x.shape}")
#         x = self.fc(x)  # [B, num_classes]
#         # print(f"After fc: {x.shape}")
#         return x


# if __name__ == "__main__":
#     from torchinfo import summary

#     # 创建模型实例
#     model = LightASF_former_S(num_classes=5)

#     # 打印模型结构
#     summary(model,
#             input_size=(1, 3, 224, 224),
#             col_names=["input_size", "output_size",
#                        "num_params", "kernel_size", "mult_adds"],
#             depth=4,
#             device="cpu")

#     # 计算参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel()
#                            for p in model.parameters() if p.requires_grad)
#     print(f"\n总参数量: {total_params:,}")
#     print(f"可训练参数量: {trainable_params:,}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class ASF_Block(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=3., qkv_bias=False, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # 自注意力
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        x = self.norm3(x)
        return x


class T2T_Module(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=384, token_dim=64):
        super().__init__()
        self.soft_split1 = nn.Unfold(kernel_size=(
            7, 7), stride=(4, 4), padding=(2, 2))
        self.soft_split2 = nn.Unfold(kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1))

        # 简化的特征提取
        self.proj1 = nn.Sequential(
            nn.Linear(in_chans * 7 * 7, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU()
        )
        self.proj2 = nn.Sequential(
            nn.Linear(token_dim * 3 * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        self.num_patches = (img_size // (4 * 2)) * (img_size // (4 * 2))

    def forward(self, x):
        # 第一次特征提取
        x = self.soft_split1(x).transpose(1, 2)
        x = self.proj1(x)

        # 重塑并第二次特征提取
        B, HW, C = x.shape
        H = W = int((HW) ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.soft_split2(x).transpose(1, 2)
        x = self.proj2(x)

        return x


class LightASF_former_S(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=5, embed_dim=384, depth=7,
                 num_heads=12, mlp_ratio=3., qkv_bias=False, drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # Token-to-Token模块
        self.tokens_to_token = T2T_Module(
            img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        self.t2t_norm = nn.LayerNorm(embed_dim)

        # 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.tokens_to_token.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ASF_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)
        x = self.t2t_norm(x)

        # 添加分类token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # 只返回分类token的特征

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# 测试代码


def test_double_branch_asfnet():
    model = LightASF_former_S(img_size=224, in_chans=3,
                              num_classes=4, embed_dim=384, depth=2)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print('Output shape:', out.shape)
    assert out.shape == (2, 4)
    print('DoubleBranchASFNet test passed!')


if __name__ == '__main__':
    test_double_branch_asfnet()
