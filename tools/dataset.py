# -*- coding: utf-8 -*-
"""
# @file name  : dataset.py
# @brief      : 数据集Dataset定义
"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, label_dir, mode='train', img_transform=None, label_transform=None):
        assert (os.path.exists(data_dir)), f"data_dir:{data_dir} 不存在！"

        self.data_dir = data_dir
        self.label_dir = label_dir
        self.mode = mode
        self._get_img_info()
        self.img_transform = img_transform

    def __getitem__(self, index):

        img_path, label = self.img_info[index]

        img = Image.open(img_path).convert("RGB")  # RGB图
        img = np.array(img)  # 将 PIL 图像转换为 numpy 数组，以便 albumentations 处理

        if self.img_transform:
            # 使用命名参数传递图像
            transformed = self.img_transform(image=img)
            img = transformed['image']

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("未获取任何图片路径，请检查dataset及文件路径！")
        return len(self.img_info)

    def _get_img_info(self):
        """
        获取图像路径和目标图像路径
        :return:
        """

        # 读取标签文件
        label_df = pd.read_csv(self.label_dir)

        # 图像根路径
        images_dir = os.path.join(self.data_dir, self.mode)
        assert os.path.exists(images_dir), f"{images_dir} 不存在！"

        self.img_info = []

        # 根据标签文件的信息创建图像路径和标签的配对
        for _, row in label_df.iterrows():
            img_id = row['图片名']
            label = row['标签']
            img_path = os.path.join(images_dir, img_id)
            if self.mode in img_path and os.path.exists(img_path):
                self.img_info.append((img_path, label))
            # else:
            #     print(f"警告: 图像路径 {img_path} 不存在或不属于当前模式 {self.mode}，已跳过。")
