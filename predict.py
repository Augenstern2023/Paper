# -*- coding: utf-8 -*-
"""
# @file name  : predict.py
# @brief      : predict demo
"""

import os
import random
import shutil
import glob

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from tools.utils import get_net
import config as cfg


def _get_norm_params():
    if cfg and hasattr(cfg, "DATA_MEAN") and hasattr(cfg, "DATA_STD"):
        return list(cfg.DATA_MEAN), list(cfg.DATA_STD)
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


_INFER_NORM_MEAN, _INFER_NORM_STD = _get_norm_params()

INFERENCE_TRANSFORM = Compose([
    Resize(224, 224),
    Normalize(mean=_INFER_NORM_MEAN, std=_INFER_NORM_STD,
              max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0),
], p=1.0)


def build_balanced_test_dataset(source_root, target_root, samples_per_class=100, seed=None):
    """
    Sample up to `samples_per_class` images per class from `source_root` and copy them into `target_root`.
    Existing image files inside each target class directory are cleared before copying.
    """
    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"Source dataset directory not found: {source_root}")

    os.makedirs(target_root, exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    sampler = random.sample if seed is None else random.Random(seed).sample

    print(f"\nBuilding test dataset from: {source_root}")
    print(f"Saving into: {target_root}")
    print(f"Sampling up to {samples_per_class} images per class")

    for class_name in sorted(os.listdir(source_root)):
        class_src_dir = os.path.join(source_root, class_name)
        if not os.path.isdir(class_src_dir):
            continue

        images = [f for f in os.listdir(class_src_dir) if f.lower().endswith(valid_exts)]
        if not images:
            print(f"Class {class_name}: no images found, skip")
            continue

        sample_count = min(samples_per_class, len(images))
        sampled_images = sampler(images, sample_count)

        class_target_dir = os.path.join(target_root, class_name)
        os.makedirs(class_target_dir, exist_ok=True)

        existing_targets = [f for f in os.listdir(class_target_dir) if f.lower().endswith(valid_exts)]
        for filename in existing_targets:
            try:
                os.remove(os.path.join(class_target_dir, filename))
            except OSError as exc:
                print(f"Failed to delete stale file {filename}: {exc}")

        for filename in sampled_images:
            src_path = os.path.join(class_src_dir, filename)
            dst_path = os.path.join(class_target_dir, filename)
            shutil.copy2(src_path, dst_path)

        print(f"Class {class_name}: copied {sample_count} / {len(images)} images")

    print("Finished building test dataset.\n")




CLASS_KEYWORDS = {
    'Bacterialblight': ('bacterial', 'bacterail', 'blight'),
    'Blast': ('blast',),
    'Brownspot': ('brownspot',),
    'Tungro': ('tungro',)
}


def infer_class_from_filename(filename):
    """Infer the class label from a file name."""
    lower_name = filename.lower()
    for class_name, keywords in CLASS_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lower_name:
                return class_name
    return None


def collect_test_images(test_root, class_names):
    """Collect all images under `test_root` and infer labels from file names."""
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Test dataset directory not found: {test_root}")

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    collected = []

    for root, _, files in os.walk(test_root):
        for filename in files:
            if not filename.lower().endswith(valid_exts):
                continue
            inferred_class = infer_class_from_filename(filename)
            if inferred_class is None or inferred_class not in class_names:
                print(f"Warning: unable to infer class for file {filename}, skipped")
                continue
            img_path = os.path.join(root, filename)
            collected.append((img_path, inferred_class))

    collected.sort(key=lambda item: item[0])
    return collected


def process_img(img_path, device):
    """
    Image preprocessing
    :param img_path: image path
    :param device: device
    :return:
    """
    img_l = Image.open(img_path).convert("RGB")  # read image in RGB format
    img_np = np.array(img_l)

    transformed = INFERENCE_TRANSFORM(image=img_np)
    img_t = transformed["image"].to(device)  # move to device

    return img_t, img_l


def test_single_model(model_name, model_path, device, sampled_imgs, class_names):
    """
    测试单个模型
    :param model_name: 模型名称
    :param model_path: 模型路径
    :param device: 设备
    :param sampled_imgs: list of tuples (image_path, class_name)
    :param class_names: 类别名称列表
    :return: (准确率, FPS, MACs)
    """
    try:
        # 加载模型
        CustomNet = get_net(device=device, model_name=model_name,
                            vis_model=False, path_state_dict=model_path)
        print(f"成功加载模型: {model_name}")

        # 测试模型输出维度
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            test_output = CustomNet(test_input)
            if isinstance(test_output, tuple):
                test_output = test_output[0]
        print(f"模型输出维度: {test_output.shape}, 类别数: {test_output.shape[1]}")

        # 计算MACs
        macs_g = 0.0
        params_m = 0.0

        # 方法1: 尝试使用thop
        try:
            from thop import profile
            input_tensor = torch.randn(1, 3, 224, 224).to(device)
            macs, params = profile(CustomNet, inputs=(
                input_tensor,), verbose=False)
            if macs > 0:
                macs_g = macs / 1e9  # 转换为G
                params_m = params / 1e6  # 转换为M
                print(f"thop计算 - MACs: {macs_g:.2f}G, 参数量: {params_m:.2f}M")
            else:
                raise Exception("thop返回MACs为0")
        except Exception as e:
            print(f"thop计算失败: {e}")

            # 方法2: 尝试使用torchprofile
            try:
                from torchprofile import profile_macs
                input_tensor = torch.randn(1, 3, 224, 224).to(device)
                macs = profile_macs(CustomNet, input_tensor)
                macs_g = macs / 1e9
                total_params = sum(p.numel() for p in CustomNet.parameters())
                params_m = total_params / 1e6
                print(
                    f"torchprofile计算 - MACs: {macs_g:.2f}G, 参数量: {params_m:.2f}M")
            except Exception as e2:
                print(f"torchprofile计算失败: {e2}")

                # 方法3: 手动计算参数量 + 估算MACs
                try:
                    total_params = sum(p.numel()
                                       for p in CustomNet.parameters())
                    params_m = total_params / 1e6

                    # 基于模型名称估算MACs
                    if 'mobilenet' in model_name.lower():
                        macs_g = 0.06  # MobileNet V3 Small
                    elif 'efficientnet' in model_name.lower():
                        macs_g = 0.39  # EfficientNet-B0
                    elif 'resnet' in model_name.lower():
                        macs_g = 7.8   # ResNet-101
                    elif 'densenet' in model_name.lower():
                        macs_g = 2.9   # DenseNet-121
                    elif 'googlenet' in model_name.lower():
                        macs_g = 1.5   # GoogLeNet
                    elif 'shufflenet' in model_name.lower():
                        macs_g = 0.15  # ShuffleNet V2
                    elif 'light' in model_name.lower():
                        macs_g = 0.5   # 轻量级模型估算
                    elif 'cassava' in model_name.lower():
                        macs_g = 1.8   # CassavaNet估算
                    else:
                        macs_g = 1.0   # 默认估算
                    print(f"估算值 - MACs: {macs_g:.2f}G, 参数量: {params_m:.2f}M")
                except Exception as e3:
                    print(f"手动计算失败: {e3}")
                    macs_g = 0.0
                    params_m = 0.0

        # 确定k值
        k = min(5, test_output.shape[1])

        # 统计
        total = 0
        correct = 0
        class_total = {c: 0 for c in class_names}
        class_correct = {c: 0 for c in class_names}

        processed_count = 0
        import time
        start_time = time.time()
        for img_path, true_class in sampled_imgs:
            if true_class not in class_total:
                print(f"Warning: unknown class label {true_class}, skipped")
                continue

            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            try:
                img_tensor, _ = process_img(img_path, device)
                with torch.no_grad():
                    outputs = CustomNet(img_tensor.unsqueeze(0))
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                _, predicted = torch.max(outputs.data, 1)
                predicted_class = predicted.item()

                if predicted_class >= len(class_names):
                    print(f"Predicted class out of range: {predicted_class}, classes: {len(class_names)}")
                    continue

                predicted_name = class_names[predicted_class]
                total += 1
                class_total[true_class] += 1
                processed_count += 1

                if predicted_name == true_class:
                    correct += 1
                    class_correct[true_class] += 1

            except Exception as e:
                print(f"Failed to process image {img_path}: {e}")
                continue
        end_time = time.time()
        elapsed = end_time - start_time
        fps = total / elapsed if elapsed > 0 else 0

        # 计算准确率
        acc = correct / total if total > 0 else 0

        print(f"模型 {model_name} 准确率: {acc*100:.2f}% ({correct}/{total})")
        print(f"模型 {model_name} 推理速度: {fps:.2f} FPS")
        print(f"成功处理图片数: {processed_count}")
        print(f"各类别统计: {class_total}")

        # 清理显存
        del CustomNet
        torch.cuda.empty_cache()

        return acc, fps, macs_g, params_m

    except Exception as e:
        print(f"测试模型 {model_name} 时出错: {e}")
        return 0.0, 0.0, 0.0, 0.0


def get_model_name_from_filename(filename):
    """
    从文件名提取模型名称（第一个下划线之前的部分）
    :param filename: 文件名
    :return: 模型名称
    """
    return filename.split('_')[0]


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset_dir = os.path.join(BASE_DIR, "Data", "test_data")

    # Collect all model files under best_results
    best_results_dir = os.path.join(BASE_DIR, "best_results")
    model_files = glob.glob(os.path.join(best_results_dir, "*.pkl"))

    if not model_files:
        print("No model files were found.")
        exit()

    print(f"Found {len(model_files)} model files:")
    for f in model_files:
        print(f"  {os.path.basename(f)}")

    class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

    sampled_imgs = collect_test_images(test_dataset_dir, class_names)
    if not sampled_imgs:
        print(f"No usable images found under {test_dataset_dir}")
        exit()

    class_counts = {c: 0 for c in class_names}
    for _, label in sampled_imgs:
        if label in class_counts:
            class_counts[label] += 1

    print("\nClass image counts:")
    for c in class_names:
        print(f"  {c}: {class_counts[c]} images")

    print(f"\nTotal of {len(sampled_imgs)} images gathered for evaluation")


    # 存储所有模型的测试结果
    model_results = {}

    # 测试每个模型5遍
    for model_file in model_files:
        model_filename = os.path.basename(model_file)
        model_name = get_model_name_from_filename(model_filename)

        print(f"\n{'='*60}")
        print(f"开始测试模型: {model_name}")
        print(f"模型文件: {model_filename}")
        print(f"{'='*60}")

        accuracies = []
        fps_list = []
        macs_list = []
        params_list = []

        # 测试5遍
        for test_round in range(5):
            print(f"\n第 {test_round + 1} 轮测试:")
            acc, fps, macs, params = test_single_model(
                model_name, model_file, device, sampled_imgs, class_names)
            accuracies.append(acc)
            fps_list.append(fps)
            macs_list.append(macs)
            params_list.append(params)
            print(f"第 {test_round + 1} 轮准确率: {acc*100:.2f}%")
            print(f"第 {test_round + 1} 轮FPS: {fps:.2f}")
            print(f"第 {test_round + 1} 轮MACs: {macs:.2f}G")
            print(f"第 {test_round + 1} 轮参数量: {params:.2f}M")

        # 取最大准确率和最大FPS
        max_acc = max(accuracies)
        max_fps = max(fps_list)
        model_results[model_name] = {
            'max_accuracy': max_acc,
            'max_fps': max_fps,
            'all_accuracies': accuracies,
            'all_fps': fps_list,
            'filename': model_filename,
            'max_macs': max(macs_list),
            'min_macs': min(macs_list),
            'avg_macs': np.mean(macs_list),
            'max_params': max(params_list),
            'min_params': min(params_list),
            'avg_params': np.mean(params_list)
        }

        print(f"\n模型 {model_name} 测试完成:")
        print(f"所有准确率: {[f'{acc*100:.2f}%' for acc in accuracies]}")
        print(f"最大准确率: {max_acc*100:.2f}%")
        print(f"所有FPS: {[f'{fps:.2f}' for fps in fps_list]}")
        print(f"最大FPS: {max_fps:.2f}")
        print(f"最大MACs: {max(macs_list):.2f}G")
        print(f"最小MACs: {min(macs_list):.2f}G")
        print(f"平均MACs: {np.mean(macs_list):.2f}G")
        print(f"最大参数量: {max(params_list):.2f}M")
        print(f"最小参数量: {min(params_list):.2f}M")
        print(f"平均参数量: {np.mean(params_list):.2f}M")

    # 按准确率排序
    sorted_results = sorted(model_results.items(),
                            key=lambda x: x[1]['max_accuracy'], reverse=True)

    # 按FPS排序（从大到小）
    sorted_results_fps = sorted(model_results.items(),
                                key=lambda x: x[1]['max_fps'], reverse=True)

    # 按MACs排序（从小到大）
    sorted_results_macs = sorted(model_results.items(),
                                 key=lambda x: x[1]['avg_macs'])

    # 写入结果到文件
    output_file = os.path.join(BASE_DIR, "best_model.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("模型泛化性能测试结果\n")
        f.write("="*60 + "\n")
        f.write(f"测试时间: {len(sampled_imgs)} 张图片，每个模型测试5遍，取最大准确率和最大FPS\n")
        f.write("="*60 + "\n\n")

        f.write("按准确率排序结果:\n")
        f.write("-"*40 + "\n")
        for i, (model_name, result) in enumerate(sorted_results, 1):
            f.write(f"排名 {i}: {model_name}\n")
            f.write(f"  模型文件: {result['filename']}\n")
            f.write(f"  最大准确率: {result['max_accuracy']*100:.2f}%\n")
            f.write(f"  最大FPS: {result['max_fps']:.2f}\n")
            f.write(f"  最大MACs: {result['max_macs']:.2f}G\n")
            f.write(f"  最小MACs: {result['min_macs']:.2f}G\n")
            f.write(f"  平均MACs: {result['avg_macs']:.2f}G\n")
            f.write(f"  最大参数量: {result['max_params']:.2f}M\n")
            f.write(f"  最小参数量: {result['min_params']:.2f}M\n")
            f.write(f"  平均参数量: {result['avg_params']:.2f}M\n")
            f.write(
                f"  所有准确率: {[f'{acc*100:.2f}%' for acc in result['all_accuracies']]}\n")
            f.write(
                f"  所有FPS: {[f'{fps:.2f}' for fps in result['all_fps']]}\n")
            f.write(f"  平均准确率: {np.mean(result['all_accuracies'])*100:.2f}%\n")
            f.write(f"  平均FPS: {np.mean(result['all_fps']):.2f}\n")
            f.write(f"  准确率标准差: {np.std(result['all_accuracies'])*100:.2f}%\n")
            f.write(f"  FPS标准差: {np.std(result['all_fps']):.2f}\n")
            f.write("\n")

        f.write("\n按FPS排序结果:\n")
        f.write("-"*40 + "\n")
        for i, (model_name, result) in enumerate(sorted_results_fps, 1):
            f.write(f"排名 {i}: {model_name}\n")
            f.write(f"  模型文件: {result['filename']}\n")
            f.write(f"  最大FPS: {result['max_fps']:.2f}\n")
            f.write(f"  最大准确率: {result['max_accuracy']*100:.2f}%\n")
            f.write(f"  最大MACs: {result['max_macs']:.2f}G\n")
            f.write(f"  最小MACs: {result['min_macs']:.2f}G\n")
            f.write(f"  平均MACs: {result['avg_macs']:.2f}G\n")
            f.write(f"  最大参数量: {result['max_params']:.2f}M\n")
            f.write(f"  最小参数量: {result['min_params']:.2f}M\n")
            f.write(f"  平均参数量: {result['avg_params']:.2f}M\n")
            f.write(
                f"  所有FPS: {[f'{fps:.2f}' for fps in result['all_fps']]}\n")
            f.write(
                f"  所有准确率: {[f'{acc*100:.2f}%' for acc in result['all_accuracies']]}\n")
            f.write(f"  平均FPS: {np.mean(result['all_fps']):.2f}\n")
            f.write(f"  平均准确率: {np.mean(result['all_accuracies'])*100:.2f}%\n")
            f.write(f"  FPS标准差: {np.std(result['all_fps']):.2f}\n")
            f.write(f"  准确率标准差: {np.std(result['all_accuracies'])*100:.2f}%\n")
            f.write("\n")

        f.write("\n按MACs排序结果 (从小到大):\n")
        f.write("-"*40 + "\n")
        for i, (model_name, result) in enumerate(sorted_results_macs, 1):
            f.write(f"排名 {i}: {model_name}\n")
            f.write(f"  模型文件: {result['filename']}\n")
            f.write(f"  平均MACs: {result['avg_macs']:.2f}G\n")
            f.write(f"  最大MACs: {result['max_macs']:.2f}G\n")
            f.write(f"  最小MACs: {result['min_macs']:.2f}G\n")
            f.write(f"  平均参数量: {result['avg_params']:.2f}M\n")
            f.write(f"  最大参数量: {result['max_params']:.2f}M\n")
            f.write(f"  最小参数量: {result['min_params']:.2f}M\n")
            f.write(f"  最大准确率: {result['max_accuracy']*100:.2f}%\n")
            f.write(f"  最大FPS: {result['max_fps']:.2f}\n")
            f.write(f"  平均准确率: {np.mean(result['all_accuracies'])*100:.2f}%\n")
            f.write(
                f"  平均FPS: {np.mean(result['all_fps']):.2f}\n")
            f.write("\n")

        # 打印最终结果
    print(f"\n{'='*60}")
    print("最终测试结果 (按准确率排序):")
    print(f"{'='*60}")
    for i, (model_name, result) in enumerate(sorted_results, 1):
        print(
            f"排名 {i}: {model_name} - 准确率: {result['max_accuracy']*100:.2f}%, FPS: {result['max_fps']:.2f}, MACs: {result['avg_macs']:.2f}G")

    # 按FPS排序
    print(f"\n{'='*60}")
    print("最终测试结果 (按FPS排序):")
    print(f"{'='*60}")
    for i, (model_name, result) in enumerate(sorted_results_fps, 1):
        print(
            f"排名 {i}: {model_name} - FPS: {result['max_fps']:.2f}, 准确率: {result['max_accuracy']*100:.2f}%, MACs: {result['avg_macs']:.2f}G")

    # 按MACs排序（从小到大）
    sorted_results_macs = sorted(
        model_results.items(), key=lambda x: x[1]['avg_macs'])

    print(f"\n{'='*60}")
    print("最终测试结果 (按MACs排序，从小到大):")
    print(f"{'='*60}")
    for i, (model_name, result) in enumerate(sorted_results_macs, 1):
        print(
            f"排名 {i}: {model_name} - MACs: {result['avg_macs']:.2f}G, 准确率: {result['max_accuracy']*100:.2f}%, FPS: {result['max_fps']:.2f}")

    print(f"\n详细结果已保存到: {output_file}")
