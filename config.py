import os
from datetime import datetime
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 基础路径
DATA_DIR = os.path.join(BASE_DIR, "processed_data")  # 数据集路径
LABEL_DIR = os.path.join(BASE_DIR, "processed_data", "train.csv")  # 数据集路径
DATA_MEAN = [0.3067036, 0.53181905, 0.44437334]    # 均值
DATA_STD = [0.16198367, 0.16350166, 0.17616951]     # 标准差

# MODEL_PATH = os.path.join(BASE_DIR, "model", "CheapNet_G64.pkl")  # 预训练模型路径
MODEL_NAME = "light_asf_net"
MODEL_PATH = None
SAVE_MODEL = True
RESUME_TRAINING = False

MAX_EPOCH = 50          # 跑多少轮
BATCH_SIZE = 8          # 每次载入多少图片
DATALOADER_WORKERS = 6  # dataloader线程数
DATA_SPLIT = 1
TIME_STR = datetime.strftime(datetime.now(), '%m-%d-%H-%M')  # 时间格式化

# ASF-former网络推荐的参数
LR = 0.001               # 学习率
MILESTONES = [10, 20, 30]     # 学习率在第多少个epoch下降
GAMMA = 0.1             # 下降参数
WEIGHT_DECAY = 1e-4

TAG = "light_asf_net_v3"      # 使用cassavanet模型
LOG_DIR = os.path.join(
    BASE_DIR, "results", f"{TAG}_P{MAX_EPOCH}_B{BATCH_SIZE}_{TIME_STR}")  # 结果保存路径
log_name = f'{TIME_STR}.log'
SAVE_MODEL_NUM = 5  # 保存效果最好的模型数量

# TensorBoard设置
ENABLE_MODEL_GRAPH = False  # 是否启用模型图可视化（设为False避免BatchNorm问题）

# RESUME_TRAINING = True  # 设置为True启用继续训练
# CHECKPOINT_PATH = os.path.join(
#     BASE_DIR, "results", "light_asf_net_P50_B8_06-25-22-32", "light_asf_net_13e_87.1069%.pkl")  # 设置检查点文件路径
