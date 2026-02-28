import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import re
import csv

# 获取基础路径（脚本所在目录的父目录）
BASE_PATH = Path(__file__).parent.parent

# 定义相对路径
Dataset_Path = BASE_PATH / 'Data' / 'Rice Leaf Disease Images'
target_data_path = BASE_PATH / 'processed_data'

# Configuration


class Config:
    # 基础路径配置
    BASE_PATH = BASE_PATH
    DATASET_PATH = Dataset_Path
    TARGET_DATA_PATH = target_data_path

    # 数据集划分配置
    VAL_FRACTION = 0.2
    RANDOM_SEED = 42

    # 类别不平衡阈值
    IMBALANCE_THRESHOLD = 0.1


def create_directories(target_path):
    """创建训练集和验证集目录"""
    (target_path / 'train').mkdir(parents=True, exist_ok=True)
    (target_path / 'val').mkdir(parents=True, exist_ok=True)

    # 为每个类别创建子目录
    for split in ['train', 'val']:
        for disease_type in get_disease_types():
            (target_path / split / disease_type).mkdir(parents=True, exist_ok=True)


def get_disease_types():
    """获取所有病害类型"""
    disease_types = []
    for item in os.listdir(Config.DATASET_PATH):
        if os.path.isdir(Config.DATASET_PATH / item):
            disease_types.append(item)
    return disease_types


def load_and_categorize_files(dataset_path):
    """按病害类型分类文件"""
    categorized_files = {}
    total_files = 0

    print("\n开始加载文件...")
    for disease_type in get_disease_types():
        disease_path = dataset_path / disease_type
        if os.path.isdir(disease_path):
            files = [f for f in os.listdir(
                disease_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            categorized_files[disease_type] = files
            total_files += len(files)
            print(f"{disease_type}: {len(files)} 张图片")

    print(f"\n总共加载了 {total_files} 张图片")
    return categorized_files


def split_dataset(categorized_files, val_frac):
    """将数据集划分为训练集和验证集"""
    train_files = {}
    val_files = {}
    total_train = 0
    total_val = 0

    print("\n开始划分数据集...")
    for disease_type, files in categorized_files.items():
        random.shuffle(files)
        val_size = int(len(files) * val_frac)

        val_files[disease_type] = files[:val_size]
        train_files[disease_type] = files[val_size:]

        total_train += len(train_files[disease_type])
        total_val += len(val_files[disease_type])
        print(f"{disease_type}:")
        print(f"  训练集: {len(train_files[disease_type])} 张")
        print(f"  验证集: {len(val_files[disease_type])} 张")

    print(f"\n划分完成:")
    print(f"训练集总数: {total_train} 张")
    print(f"验证集总数: {total_val} 张")
    print(f"总图片数: {total_train + total_val} 张")

    return train_files, val_files


def print_statistics(categorized_files, train_files, val_files):
    """打印数据集统计信息"""
    print('\n数据集统计信息:')
    print('总类别数:', len(categorized_files))

    print('\n训练集:')
    for disease_type, files in train_files.items():
        print(f'{disease_type}: {len(files)} 张图片')

    print('\n验证集:')
    for disease_type, files in val_files.items():
        print(f'{disease_type}: {len(files)} 张图片')


def copy_files(files_dict, src_path, dst_path, desc):
    """复制文件到目标目录，不保留原来的目录结构"""
    total_copied = 0
    for disease_type, files in files_dict.items():
        src_disease_path = src_path / disease_type
        for file in tqdm(files, desc=f"{desc} - {disease_type}"):
            shutil.copy(src_disease_path / file, dst_path / file)
            total_copied += 1
    print(f"\n{desc}完成，共复制 {total_copied} 张图片")


def check_class_balance(train_files):
    """检查并报告类别平衡情况"""
    total_samples = sum(len(files) for files in train_files.values())
    print("\n训练集类别分布:")

    for disease_type, files in train_files.items():
        ratio = len(files) / total_samples
        print(f"{disease_type}: {ratio:.2%}")

    # 检查类别不平衡
    ratios = [len(files) / total_samples for files in train_files.values()]
    max_ratio = max(ratios)
    min_ratio = min(ratios)

    if max_ratio - min_ratio > Config.IMBALANCE_THRESHOLD:
        print("\n注意: 训练集中存在类别不平衡")
        print("建议在训练时使用类别权重:")

        class_weights = {
            disease_type: total_samples / (len(files) * len(train_files))
            for disease_type, files in train_files.items()
        }

        print("类别权重:", class_weights)


def generate_train_csv(train_dir, val_dir, output_csv):
    """生成train.csv标签文件，包含训练集和验证集的图片"""
    # 类别与标签的映射（使用小写作为键）
    label_map = {
        'bacterialblight': '0',
        'blast': '1',
        'brownspot': '2',
        'tungro': '3',
    }

    # 处理拼写错误的映射
    spelling_map = {
        'bacterailblight': 'bacterialblight',  # 处理拼写错误
    }

    # 正则表达式匹配类别（不区分大小写）
    pattern = re.compile(
        r'(bacterailblight|bacterialblight|blast|brownspot|tungro)', re.IGNORECASE)

    rows = []
    total_files = 0
    processed_files = 0

    print("\n开始生成train.csv...")
    for directory in [train_dir, val_dir]:
        print(f"\n处理目录: {directory}")
        for file in os.listdir(directory):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_files += 1
                match = pattern.search(file.lower())  # 转换为小写进行匹配
                if match:
                    disease = match.group(1).lower()  # 获取匹配的类别并转换为小写
                    # 处理拼写错误
                    disease = spelling_map.get(disease, disease)
                    label = label_map.get(disease)
                    if label is not None:
                        rows.append([file, label])
                        processed_files += 1
                    else:
                        print(f"警告: 无法找到类别标签 - {file}")
                else:
                    print(f"警告: 无法匹配类别 - {file}")

    print(f"\n文件处理统计:")
    print(f"总文件数: {total_files}")
    print(f"成功处理文件数: {processed_files}")
    print(f"未处理文件数: {total_files - processed_files}")

    # 写入csv
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['图片名', '标签'])
        writer.writerows(rows)
    print(f"\ntrain.csv标签文件已生成，共{len(rows)}条，保存于: {output_csv}")


def main():
    """主函数：执行数据集划分过程"""
    # # 设置随机种子以确保可重复性
    # random.seed(Config.RANDOM_SEED)

    # # 创建必要的目录
    # create_directories(Config.TARGET_DATA_PATH)

    # # 加载并分类文件
    # categorized_files = load_and_categorize_files(Config.DATASET_PATH)

    # # 划分数据集
    # train_files, val_files = split_dataset(
    #     categorized_files, Config.VAL_FRACTION)

    # # 打印统计信息
    # print_statistics(categorized_files, train_files, val_files)

    # # 复制文件
    # copy_files(train_files, Config.DATASET_PATH,
    #            Config.TARGET_DATA_PATH / 'train', "复制训练集文件")
    # copy_files(val_files, Config.DATASET_PATH,
    #            Config.TARGET_DATA_PATH / 'val', "复制验证集文件")

    # # 检查类别平衡
    # check_class_balance(train_files)

    # 生成train.csv标签文件
    train_dir = Config.TARGET_DATA_PATH / 'train'
    val_dir = Config.TARGET_DATA_PATH / 'val'
    output_csv = BASE_PATH / 'processed_data' / 'train.csv'
    generate_train_csv(train_dir, val_dir, output_csv)


if __name__ == "__main__":
    main()
