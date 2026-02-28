# -*- coding: utf-8 -*-
"""
# @file name  : utils.py
# @brief      : 通用函数
"""
from net.light_asf_net import LightASF_former_S
import torch.nn.functional as F
from net.resnet import ResNet, BasicBlock
import cv2
import numpy as np
import torchvision
from PIL import Image

from net.Baseline_nograd import Baseline
import torch
import torch.nn as nn
import os
import logging
import sys
import torchmetrics
import pandas as pd

from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 将ASF-former模型的导入移动到get_net函数内部


def clear_cuda_cache():
    torch.cuda.empty_cache()


def get_model(model_name, num_classes=4, weights=None):
    if model_name == "resnet101":
        model = models.resnet101(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "googlenet":
        model = models.googlenet(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "mobilenet":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, num_classes)
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet":
        # 导入EfficientNet-B0模型
        from net.EfficientNet import efficientnet_b0
        model = efficientnet_b0(num_classes=num_classes)
    # elif model_name == "shufflenet":
    #     # 导入ShuffleNet模型
    #     from net.shufflenet import shufflenet
    #     model = shufflenet()
    #     # 修改最后的全连接层以匹配类别数
    #     model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "cassavanet":
        # 导入CassavaNet模型
        from net.CassavaNet import CassavaNet
        model = CassavaNet(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return model


# 直接支持LightASF_former_S网络


def get_net(device, model_name="Baseline", vis_model=False, path_state_dict=None):
    """
    创建模型，加载参数
    :param device: 运算设备
    :param vis_model: 是否打印模型结构
    :param path_state_dict:
    :return: 预训练模型
    """
    if model_name == "Baseline":
        from net.Baseline_nograd import Baseline
        model = Baseline()
    elif model_name == "light":
        model = LightASF_former_S(
            num_classes=4, img_size=224, in_chans=3, depth=7)
    else:
        model = get_model(model_name, num_classes=4, weights=None)  # 不加载预训练权重

    if path_state_dict:
        pretrained_state_dict = torch.load(
            path_state_dict, map_location=device)  # 读取预训练模型
        model.load_state_dict(pretrained_state_dict['CustomNet'])  # 将预训练模型载入模型

    model.eval()  # 开启验证模式

    if vis_model:  # 是否打印模型结构
        from torchinfo import summary
        summary(model, input_size=(1, 3, 200, 200), device="cpu")

    model.to(device)  # 将模型推至运算设备
    return model


class CustomNetTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch, logger):
        """
        每次传入一个epoch的数据进行模型训练
        :param data_loader: 训练集加载器
        :param model: 模型
        :param loss_f: 损失函数
        :param optimizer: 优化器
        :param epoch_id: 第几个epoch
        :param device: 运算设备
        :param max_epoch: 最大训练轮数
        :param logger: 日志
        :return: 平均loss
        """
        model.train()  # 开启模型训练模式

        loss_avg = []  # 平均loss
        for i, data in enumerate(data_loader):  # 迭代训练集加载器,得到iteration和相关图像data

            x, target = data  # 通过data得到图像数据
            x = x.to(device)  # 传入运算设备
            target = target.to(device)  # 传入运算设备

            y = model(x)  # 载入模型,得到预测值

            optimizer.zero_grad()  # 优化器梯度归零
            loss = loss_f(y, target)  # 计算每个预测值与target的损失
            loss.backward()  # 反向传播,计算梯度
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()  # 更新梯度

            loss_avg.append(loss.item())  # 记录每次的loss值

            logger.info(f'Train | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] '
                        f'Iteration[{i + 1:0>3}/{len(data_loader):0>3}] '
                        f'Train loss: {np.mean(loss_avg):.8f}')

        # 在训练开始前和每个epoch后调用
        clear_cuda_cache()

        return np.mean(loss_avg)

    @staticmethod
    def valid(data_loader, model, loss_f, epoch_id, device, max_epoch, logger):
        """
        模型验证
        :param data_loader: 验证集加载器
        :param model: 模型
        :param loss_f: 损失函数
        :param epoch_id: 第几个epoch
        :param device: 运算设备
        :param max_epoch: 最大训练轮数
        :param logger: 日志
        :return: 平均loss
        """
        model.eval()  # 模型验证模式
        accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=4).to(device)
        precision = torchmetrics.Precision(
            task="multiclass", num_classes=4).to(device)
        recall = torchmetrics.Recall(
            task="multiclass", num_classes=4).to(device)
        specificity = torchmetrics.Specificity(
            task="multiclass", num_classes=4).to(device)
        confusion_matrix = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=4).to(device)

        class_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=4, average=None).to(device)
        class_precision = torchmetrics.Precision(
            task="multiclass", num_classes=4, average=None).to(device)
        class_recall = torchmetrics.Recall(
            task="multiclass", num_classes=4, average=None).to(device)
        class_specificity = torchmetrics.Specificity(
            task="multiclass", num_classes=4, average=None).to(device)

        loss_avg = []  # 平均loss

        with torch.no_grad():  # 验证时不需要计算梯度，节省显存
            for i, data in enumerate(data_loader):  # 迭代验证集加载器,得到iteration和相关data
                x, target = data    # 通过data得到图像数据和对应的label
                x = x.to(device)    # 传入运算设备
                target = target.to(device)  # 传入运算设备

                # 获取模型输出
                output = model(x)

                # 计算损失
                loss = loss_f(output, target)
                loss_avg.append(loss.item())  # 记录每次的loss值

                # 统计预测信息
                _, predicted = torch.max(output.data, 1)

                accuracy.update(predicted, target)
                precision.update(predicted, target)
                recall.update(predicted, target)
                specificity.update(predicted, target)
                confusion_matrix.update(predicted, target)

                class_accuracy.update(predicted, target)
                class_precision.update(predicted, target)
                class_recall.update(predicted, target)
                class_specificity.update(predicted, target)

                logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] '
                            f'Iteration[{i + 1:0>3}/{len(data_loader):0>3}] '
                            f'Valid loss: {np.mean(loss_avg):.4f}')

                # 定期清理显存
                if i % 10 == 0 and i > 0:
                    torch.cuda.empty_cache()

        # 验证结束后清理显存
        torch.cuda.empty_cache()

        valid_acc = accuracy.compute().item()
        valid_precision = precision.compute().item()
        valid_recall = recall.compute().item()
        valid_specificity = specificity.compute().item()
        valid_conf_matrix = confusion_matrix.compute().cpu().numpy()

        valid_class_acc = class_accuracy.compute().cpu().numpy()
        valid_class_precision = class_precision.compute().cpu().numpy()
        valid_class_recall = class_recall.compute().cpu().numpy()
        valid_class_specificity = class_specificity.compute().cpu().numpy()

        valid_macc = valid_class_acc.mean()  # 计算每个类别的准确率，然后求平均值

        class_names = ['bacterialblight', 'blast',
                       'brownspot', 'tungro']  # 定义类名
        # 使用类别名称创建带标注的混淆矩阵
        df_conf_matrix = pd.DataFrame(
            valid_conf_matrix, index=class_names, columns=class_names)

        logger.info(
            "============================================================================")
        logger.info(
            f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Accuracy: {valid_acc:.4f}')
        logger.info(
            f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Mean Accuracy (mACC): {valid_macc:.4f}')
        logger.info(
            f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Precision: {valid_precision:.4f}')
        logger.info(
            f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Recall: {valid_recall:.4f}')
        logger.info(
            f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Specificity: {valid_specificity:.4f}')
        logger.info(
            f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Confusion Matrix: \n{df_conf_matrix}')
        logger.info(
            "============================================================================")
        for idx, class_name in enumerate(class_names):
            logger.info(
                f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] {class_name} Accuracy: {valid_class_acc[idx]:.4f}')
            logger.info(
                f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] {class_name} Precision: {valid_class_precision[idx]:.4f}')
            logger.info(
                f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] {class_name} Recall: {valid_class_recall[idx]:.4f}')
            logger.info(
                f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] {class_name} Specificity: {valid_class_specificity[idx]:.4f}')
        logger.info(
            "============================================================================")

        accuracy.reset()
        precision.reset()
        recall.reset()
        specificity.reset()
        confusion_matrix.reset()
        class_accuracy.reset()
        class_precision.reset()
        class_recall.reset()
        class_specificity.reset()

        return np.mean(loss_avg), valid_macc


def get_logger(log_dir, log_name):

    log_file = os.path.join(log_dir, log_name)

    # 创建log
    logger = logging.getLogger('train')  # log初始化
    logger.setLevel(logging.INFO)  # 设置log级别, INFO是程序正常运行时输出的信息

    # Formatter 设置日志输出格式
    formatter = logging.Formatter(
        '%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # StreamHandler 日志输出1 -> 输出到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler 日志输出2 -> 保存到文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
