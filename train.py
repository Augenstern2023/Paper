import os
import shutil
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as opt
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np
import warnings
from tools.dataset import CustomDataset
from tools.utils import get_net, CustomNetTrainer, get_logger
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from albumentations import (
    HorizontalFlip, VerticalFlip, Perspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine, RandomResizedCrop,
    Sharpen, Emboss, RandomBrightnessContrast, OneOf, Compose, Normalize, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import config as cfg

# 设置环境变量以优化显存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 调试用，生产环境可移除
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
warnings.filterwarnings('ignore')


def clear_cuda_cache():
    torch.cuda.empty_cache()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # 基础路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 运算设备


def get_grad_norm(model):
    """计算模型参数的梯度范数"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


if __name__ == "__main__":

    # 网络配置
    data_dir = cfg.DATA_DIR      # 基础路径
    label_dir = cfg.LABEL_DIR    # 标签路径
    max_epoch = cfg.MAX_EPOCH    # 跑多少轮
    batch_size = cfg.BATCH_SIZE  # 每次载入多少图片
    model_path = cfg.MODEL_PATH  # 预训练模型

    # 优化器配置
    lr = cfg.LR                  # 学习率
    milestones = cfg.MILESTONES  # 学习率在第多少个epoch下降
    gamma = cfg.GAMMA            # 下降参数

    # 输出结果目录
    output_dir = cfg.LOG_DIR     # 结果保存路径
    log_name = cfg.log_name      # 日志文件路径

    # 标准化参数
    norm_mean = cfg.DATA_MEAN   # 均值
    norm_std = cfg.DATA_STD     # 标准差

    # 若文件夹不存在,则创建
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=output_dir)  # 创建tensorboard文件

    if os.path.exists("config.py"):
        shutil.copy("config.py", output_dir)          # 将当前配置文件拷贝一份到输出文件夹

    logger = get_logger(output_dir, log_name)   # 创建日志文件

    logger.info(f'Start | Model starts training!!!\n')

    # ============================ step 1/5 数据 ============================

    def get_train_transforms():
        return Compose([
            RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=norm_mean, std=norm_std,
                      max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

    def get_valid_transforms():
        return Compose([
            Resize(224, 224),
            Normalize(mean=norm_mean, std=norm_std,
                      max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

    # 构建DAGMDataset
    train_data = CustomDataset(data_dir=data_dir, label_dir=label_dir,
                               mode='train', img_transform=get_train_transforms())
    valid_data = CustomDataset(data_dir=data_dir, label_dir=label_dir,
                               mode='val', img_transform=get_valid_transforms())

    # 创建训练数据的索引
    train_size = len(train_data)
    indices = list(range(train_size))
    np.random.shuffle(indices)

    # 取1/3的数据进行训练
    sample_size = train_size // cfg.DATA_SPLIT
    train_indices = indices[:sample_size]

    # 创建采样器
    train_sampler = SubsetRandomSampler(train_indices)

    # 构建DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                              sampler=train_sampler, num_workers=cfg.DATALOADER_WORKERS)
    valid_loader = DataLoader(
        dataset=valid_data, batch_size=batch_size, num_workers=cfg.DATALOADER_WORKERS)

    logger.info(
        f'Using {sample_size} samples for training (1/{cfg.DATA_SPLIT} of total data)')
    logger.info(
        f'Using full validation set with {len(valid_data)} samples')

    # ============================ step 2/5 模型 ============================

    model_name = cfg.MODEL_NAME if hasattr(
        cfg, 'MODEL_NAME') else "asf_resnet_b"  # 默认使用B模型
    custom_model = get_net(device=device, model_name=model_name,
                           vis_model=False, path_state_dict=model_path)

    # TensorBoard模型图可视化（可选）
    ENABLE_MODEL_GRAPH = getattr(cfg, 'ENABLE_MODEL_GRAPH', True)  # 默认启用

    if ENABLE_MODEL_GRAPH:
        try:
            # 使用batch_size=2的输入
            dummy_input = torch.rand(2, 3, 224, 224).to(device)
            writer.add_graph(custom_model, input_to_model=dummy_input)
            logger.info("Successfully added model graph to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to add model graph to TensorBoard: {e}")
            logger.info("Continuing without model graph visualization")
    else:
        logger.info("Model graph visualization disabled")

    # ============================ step 3/5 损失函数 ============================

    # 计算每个类别的样本数量
    # CBB, CBSD, CGM, CMD, Heath
    class_counts = np.array([1581, 1437, 1597, 1305])
    # 使用样本数量的倒数作为权重，并进行归一化
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.from_numpy(class_weights).float()
    class_weights = class_weights.to(device)

    # 使用nn.CrossEntropyLoss替代FocalLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ============================ step 4/5 优化器 ============================

    # 优化器
    optimizer = opt.AdamW(custom_model.parameters(), lr=lr, weight_decay=1e-5)
    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epoch,
        eta_min=1e-6
    )
    # ============================ step 5/6 加载预训练pkl文件 ============================
    # 如果存在checkpoint，加载模型继续训练
    start_epoch = 0
    if cfg.RESUME_TRAINING and os.path.exists(cfg.CHECKPOINT_PATH):
        logger.info(f'Loading checkpoint from {cfg.CHECKPOINT_PATH}')
        checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=device)

        # 加载模型参数
        if 'CustomNet' in checkpoint:
            custom_model.load_state_dict(checkpoint['CustomNet'])
            logger.info('Successfully loaded model parameters')

        # 加载优化器状态
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('Successfully loaded optimizer state')

        # 加载学习率调度器状态
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info('Successfully loaded scheduler state')

        # 加载训练轮数
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            logger.info(f'Resuming from epoch {start_epoch}')

        # 加载最佳模型记录
        if 'best_models' in checkpoint:
            best_models = checkpoint['best_models']
            logger.info('Successfully loaded best models record')

    logger.info(f'Using model: {model_name}')
    # ============================ step 5/5 训练 ============================

    start_time = datetime.now()          # 训练开始时间
    best_models = []                     # 用于保存效果最好的模型及其指标
    save_model_num = cfg.SAVE_MODEL_NUM  # 保存效果最好的模型数量

    # 添加梯度监控
    grad_norm_history = []
    loss_history = []
    patience = 10  # 早停耐心值
    min_loss = float('inf')
    no_improve_epochs = 0

    # 检查batch_size，确保BatchNorm正常工作
    if batch_size < 2:
        logger.warning(
            f"Batch size {batch_size} is too small for BatchNorm. Setting to 2.")
        batch_size = 2
        # 重新创建DataLoader
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                                  sampler=train_sampler, num_workers=cfg.DATALOADER_WORKERS)
        valid_loader = DataLoader(
            dataset=valid_data, batch_size=batch_size, num_workers=cfg.DATALOADER_WORKERS)

    for epoch in range(0, max_epoch):    # 模型开始训练
        # 训练阶段
        custom_model.train()
        loss_train = 0
        epoch_grad_norms = []

        for i, data in enumerate(train_loader):
            x, target = data
            x = x.to(device)
            target = target.to(device)

            # 检查batch_size，确保BatchNorm正常工作
            if x.size(0) < 2:
                logger.warning(
                    f"Skip batch with size {x.size(0)} (too small for BatchNorm)")
                continue

            # 前向传播
            optimizer.zero_grad()
            try:
                y = custom_model(x)
            except RuntimeError as e:
                if "Expected more than 1 value per channel" in str(e):
                    logger.error(f"BatchNorm error: {e}")
                    logger.error(f"Current batch size: {x.size(0)}")
                    # 清理GPU内存
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            # 计算损失
            loss = criterion(y, target)

            # 反向传播
            loss.backward()

            # 计算梯度范数
            grad_norm = get_grad_norm(custom_model)
            epoch_grad_norms.append(grad_norm)

            # # 动态梯度裁剪
            # if grad_norm > 1.0:
            #     torch.nn.utils.clip_grad_norm_(
            #         custom_model.parameters(), max_norm=1.0)

            optimizer.step()

            loss_train += loss.item()

            # # 动态学习率调整
            # if i > 0 and i % 100 == 0:
            #     avg_grad_norm = sum(epoch_grad_norms[-100:]) / 100
            #     if avg_grad_norm < 1e-4:  # 如果梯度太小
            #         for param_group in optimizer.param_groups:
            #             param_group['lr'] *= 1.1  # 增加学习率
            #     elif avg_grad_norm > 1.0:  # 如果梯度太大
            #         for param_group in optimizer.param_groups:
            #             param_group['lr'] *= 0.9  # 减小学习率

            logger.info(f'Train | Epoch[{epoch + 1:0>3}/{max_epoch:0>3}] '
                        f'Iteration[{i + 1:0>3}/{len(train_loader):0>3}] '
                        f'Train loss: {loss.item():.8f} '
                        f'Grad norm: {grad_norm:.4f} '
                        f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # 检查参数是否出现NaN或Inf
            for name, param in custom_model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"参数{name}出现NaN或Inf")

            # 定期清理GPU内存
            if i % 50 == 0:
                torch.cuda.empty_cache()

        loss_train = loss_train / len(train_loader)
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
        grad_norm_history.append(avg_grad_norm)
        loss_history.append(loss_train)

        # 验证阶段
        loss_valid, mean_acc = CustomNetTrainer.valid(
            valid_loader, custom_model, criterion, epoch, device, max_epoch, logger)

        # 早停检查
        if loss_valid < min_loss:
            min_loss = loss_valid
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(
                    f'Early stopping triggered after {epoch + 1} epochs')
                break

        # 更新学习率
        scheduler.step()

        # 记录日志
        logger.info(f'Stage | Epoch[{epoch + 1:0>3}/{max_epoch:0>3}] '
                    f'Train loss:{loss_train:.8f} '
                    f'Valid loss:{loss_valid:.8f} '
                    f'Grad norm:{avg_grad_norm:.4f} '
                    f'LR:{optimizer.param_groups[0]["lr"]:.6f}\n')

        # tensorboard记录
        writer.add_scalars(
            "Loss", {"train": loss_train, "valid": loss_valid}, epoch)
        writer.add_scalar("Grad_Norm", avg_grad_norm, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("MACC", mean_acc, epoch)

        # 保存模型
        if cfg.SAVE_MODEL:
            if len(best_models) < save_model_num or (best_models and mean_acc > best_models[-1][1]):
                checkpoint = {
                    "CustomNet": custom_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_models": best_models,
                    "grad_norm_history": grad_norm_history,
                    "loss_history": loss_history
                }
                path_checkpoint = os.path.join(
                    output_dir, f"{cfg.TAG}_{epoch+1}e_{mean_acc:.4%}.pkl")
                torch.save(checkpoint, path_checkpoint)

                best_models.append((path_checkpoint, mean_acc))
                best_models.sort(key=lambda x: x[1], reverse=True)
                if len(best_models) > save_model_num:
                    worst_model_path = best_models[-1][0]
                    os.remove(worst_model_path)
                    del best_models[-1]

    end_time = datetime.now()                     # 训练结束时间
    spend_time = (end_time - start_time).seconds  # 训练花费时间(s)
    logger.info(f'Final | Model training completed!!!')

    # logger.info(f'Final | Generating inference image video, FPS {cfg.VIDEO_FPS}')
    # images_to_video(output_dir, cfg.VIDEO_FPS)

    logger.info(
        f'Final | Start time: {datetime.strftime(start_time, "%Y-%m-%d %H:%M:%S")}')
    logger.info(
        f'Final | End time: {datetime.strftime(end_time, "%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Final | Spend time: {spend_time}s')
    logger.info(f'Final | best_mAcc: {best_models[-1][1]:.8%}')
    logger.info(f'Final | Final epoch is {max_epoch}')
    logger.info(f'Final | Each epoch spend {spend_time/max_epoch}s')

    writer.close()  # 关闭writer
