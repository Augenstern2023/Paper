from models.asf_former import ASF_former, T2T_module
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class AFFModule(nn.Module):
    def __init__(self, in_features):
        super(AFFModule, self).__init__()
        self.fc_weight = nn.Linear(in_features * 2, 2)
        nn.init.constant_(self.fc_weight.weight, 0)
        nn.init.constant_(self.fc_weight.bias, 0)

    def forward(self, local_feature, global_feature):
        # local_feature, global_feature: [B, C]
        concat = torch.cat([local_feature, global_feature], dim=1)  # [B, 2C]
        weights = self.fc_weight(concat)  # [B, 2]
        weights = F.softmax(weights, dim=1)  # [B, 2]
        local_weight = weights[:, 0:1]  # [B, 1]
        global_weight = weights[:, 1:2]  # [B, 1]
        fused = local_feature * local_weight + \
            global_feature * global_weight  # [B, C]
        return fused


class LightASF_former_S(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=10, embed_dim=384, depth=7,
                 num_heads=12, mlp_ratio=3., qkv_bias=False, drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        # 主分支1：ASF-former
        self.asf_branch = ASF_former(
            img_size=img_size,
            tokens_type='performer',
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            token_dim=64,
            ASF=True
        )
        # 主分支2：T2T_module + 简单MLP分类头
        self.t2t_module = T2T_module(
            img_size=img_size,
            tokens_type='performer',
            in_chans=in_chans,
            embed_dim=embed_dim,
            token_dim=64
        )
        self.t2t_norm = nn.LayerNorm(embed_dim)
        self.t2t_head = nn.Linear(embed_dim, num_classes)
        # AFF融合层
        self.aff_fusion = AFFModule(num_classes)

    def forward(self, x):
        # 分支1：ASF-former
        asf_out = self.asf_branch(x)  # [B, num_classes]
        # 分支2：T2T_module + MLP
        t2t_tokens, *_ = self.t2t_module(x)  # [B, N, embed_dim]
        t2t_feat = t2t_tokens.mean(dim=1)  # 全局平均池化
        t2t_feat = self.t2t_norm(t2t_feat)
        t2t_out = self.t2t_head(t2t_feat)  # [B, num_classes]
        # AFF自适应融合
        out = self.aff_fusion(t2t_out, asf_out)  # [B, num_classes]
        return out


# 测试代码
if __name__ == '__main__':
    model = LightASF_former_S(img_size=128, in_chans=3,
                              num_classes=10, embed_dim=384, depth=2)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    print('Output shape:', out.shape)
    assert out.shape == (x.shape[0], model.asf_branch.num_classes)
    print('Test passed!')
