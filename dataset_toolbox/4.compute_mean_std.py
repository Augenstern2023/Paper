# coding: utf-8
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

"""
    计算processed_data目录下所有图片的均值和标准差
    先将像素从0～255归一化至 0-1 再计算
"""

# 获取基础路径（脚本所在目录的父目录）
BASE_PATH = Path(__file__).parent.parent

# 定义数据目录
train_dir = BASE_PATH / 'processed_data' / 'train'
val_dir = BASE_PATH / 'processed_data' / 'val'

# 定义均值和方差
total_mean = 0.0
total_var = 0.0
num_images = 0

# 创建错误日志目录
error_log_dir = BASE_PATH / 'processed_data' / 'error_logs'
error_log_dir.mkdir(exist_ok=True)


def check_image(image_path):
    """检查图片是否可读"""
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            return False, "文件不存在"

        # 检查文件大小
        if os.path.getsize(image_path) == 0:
            return False, "文件大小为0"

        # 尝试读取图片
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "无法读取图片"

        # 检查图片尺寸
        if img.size == 0:
            return False, "图片尺寸为0"

        return True, img
    except Exception as e:
        return False, str(e)


def process_directory(directory):
    """处理指定目录下的所有图片"""
    global total_mean, total_var, num_images

    print(f"\n处理目录: {directory}")
    error_files = []

    for image_file in tqdm(os.listdir(directory)):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory, image_file)

            # 检查图片
            is_valid, result = check_image(image_path)

            if is_valid:
                try:
                    # 转换为浮点型数据
                    image = result.astype(np.float32) / 255.0

                    # 计算均值和方差
                    mean = np.mean(image, axis=(0, 1))
                    var = np.var(image, axis=(0, 1))

                    # 累加均值和方差
                    total_mean += mean
                    total_var += var
                    num_images += 1

                except Exception as e:
                    error_files.append((image_path, f"处理错误: {str(e)}"))
            else:
                error_files.append((image_path, result))

    # 记录错误文件
    if error_files:
        error_log_file = error_log_dir / f"{directory.name}_errors.txt"
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"目录 {directory} 中的错误文件:\n")
            for file_path, error_msg in error_files:
                f.write(f"{file_path}: {error_msg}\n")
        print(f"\n发现 {len(error_files)} 个错误文件，详细信息已保存到: {error_log_file}")


def main():
    """主函数"""
    # 处理训练集和验证集
    process_directory(train_dir)
    process_directory(val_dir)

    if num_images > 0:
        # 计算平均值
        avg_mean = total_mean / num_images
        avg_std = np.sqrt(total_var / num_images)

        print('\n计算结果:')
        print(f'成功处理的图片总数: {num_images}')
        print('均值:', avg_mean)
        print('标准差:', avg_std)

        # 保存结果到文件
        result_file = BASE_PATH / 'processed_data' / 'mean_std.txt'
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f'成功处理的图片总数: {num_images}\n')
            f.write(f'均值: {avg_mean}\n')
            f.write(f'标准差: {avg_std}\n')
        print(f'\n结果已保存到: {result_file}')
    else:
        print('没有找到任何有效图片')


if __name__ == "__main__":
    main()

# V1数据集结果
# 均值: [0.71254545 0.5873999  0.25357783]
# 标准差: [0.05730477 0.17414872 0.17775692]

# V2数据集结果
# 均值: [0.717044   0.60302377 0.26211783]
# 标准差: [0.05540495 0.17206946 0.18077734]

# ABIDEII-FisherZ数据集结果
# 均值: [0.7133571  0.73754334 0.5390207 ]
# 标准差: [0.10032479 0.2270109  0.3069458 ]
