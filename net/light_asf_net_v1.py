import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class MLP_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ASF_DoubleBranch_Block(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=3., qkv_bias=False, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.norm_split1 = nn.LayerNorm(dim//2)
        self.norm_split2 = nn.LayerNorm(dim//2)
        self.attn = nn.MultiheadAttention(
            dim//2, num_heads, dropout=attn_drop, bias=qkv_bias)
        self.mlp1 = MLP_Block(dim//2, mlp_ratio, drop)
        self.mlp2 = MLP_Block(dim//2, mlp_ratio=4., drop=drop)
        self.fusion_fc = nn.Linear(dim//2, 1)  # 用于生成融合权重
        self.sigmoid = nn.Sigmoid()
        self.expand = nn.Linear(dim//2, dim)
        self.norm_fuse = nn.LayerNorm(dim)
        self.mlp_fuse = MLP_Block(dim, mlp_ratio, drop)

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        x1, x2 = x.split(D//2, dim=2)  # 通道分裂
        # 分支1：自注意力
        y1 = self.norm_split1(x1)
        y1 = y1.transpose(0, 1)  # [N, B, D/2]
        y1 = self.attn(y1, y1, y1)[0]
        y1 = y1.transpose(0, 1)  # [B, N, D/2]
        y1 = self.mlp1(y1)
        # 分支2：无自注意力，仅MLP
        y2 = self.norm_split2(x2)
        y2 = self.mlp2(y2)
        # 融合
        S = y1 + y2  # 残差
        pool = S.mean(dim=1)  # 全局平均池化 [B, D/2]
        alpha = self.sigmoid(self.fusion_fc(pool))  # [B, 1]
        beta = 1 - alpha
        y_fused = alpha.unsqueeze(1) * y1 + beta.unsqueeze(1) * y2 + S
        # 融合后MLP+残差
        y_fused = self.expand(y_fused)  # [B, N, D]
        y_fused = self.norm_fuse(y_fused)
        y_out = self.mlp_fuse(y_fused) + y_fused
        return y_out


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
            ASF_DoubleBranch_Block(
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
                              num_classes=5, embed_dim=384, depth=2)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print('Output shape:', out.shape)
    assert out.shape == (x.shape[0], model.num_classes)
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print('DoubleBranchASFNet test passed!')


if __name__ == '__main__':
    test_double_branch_asfnet()
