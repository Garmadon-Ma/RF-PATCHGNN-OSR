import os
import itertools
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from openood.utils import config
from openood.datasets import get_dataloader
from openood.networks.resnet18_32x32 import ResNet18_32x32


# 全局绘图字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False


os.environ['PYTHONHASHSEED'] = str(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConfigFineGrained:
    def __init__(self):
        self.num_classes = 8
        self.save_dir = "results_fine_grained"
        self.device = device
        # 使用 fine-grained 数据集配置
        self.config_files = [
            "./configs/datasets/fine-grained/rml201610a.yml",
            "./configs/networks/resnet18_32x32.yml",
            "./configs/pipelines/test/test_ood.yml",
            "./configs/preprocessors/base_preprocessor.yml",
            "./configs/postprocessors/msp.yml",
        ]


def get_predictions(model: nn.Module, data_loader, device: torch.device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Getting predictions"):
            data = batch["data"].to(device)
            labels = batch["label"].to(device)

            logits = model(data)
            _, pred = torch.max(logits, dim=1)

            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    title="Confusion Matrix",
    save_path=None,
    figsize=(10, 8),
    normalize=False,
):
    cm_raw = confusion_matrix(y_true, y_pred)

    if normalize:
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        cm_display = np.divide(
            cm_raw.astype("float"),
            row_sums,
            out=np.zeros_like(cm_raw, dtype=float),
            where=(row_sums != 0),
        )
        cm_display = np.round(cm_display, 3)
    else:
        cm_display = cm_raw

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]

    plt.figure(figsize=figsize)
    im = plt.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm_display.max() / 2.0 if cm_display.size > 0 else 0.5
    for i, j in itertools.product(range(cm_display.shape[0]), range(cm_display.shape[1])):
        if normalize:
            if cm_display[i, j] > 0:
                text = f"{cm_display[i, j]:.1%}"
            else:
                text = "0%"
        else:
            text = f"{int(cm_display[i, j])}"
        plt.text(
            j,
            i,
            text,
            horizontalalignment="center",
            color="white" if cm_display[i, j] > thresh else "black",
            fontsize=12,
        )

    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"混淆矩阵已保存到: {save_path}")

    plt.show()

    accuracy = np.trace(cm_raw) / np.sum(cm_raw)
    print(f"总体准确率: {accuracy:.4f}")

    precision = np.diag(cm_raw) / np.sum(cm_raw, axis=0)
    recall = np.diag(cm_raw) / np.sum(cm_raw, axis=1)
    f1 = 2 * precision * recall / (precision + recall)

    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)

    print("\n各类别性能指标:")
    print(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    print("-" * 45)
    for i, class_name in enumerate(class_names):
        print(
            f"{class_name:<15} {precision[i]:<10.4f} "
            f"{recall[i]:<10.4f} {f1[i]:<10.4f}"
        )

    return cm_raw, accuracy, precision, recall, f1


def plot_confusion_matrix_from_model(
    model,
    data_loader,
    class_names=None,
    title="Confusion Matrix",
    save_path=None,
    figsize=(10, 8),
    normalize=False,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("正在获取预测结果...")
    y_pred, y_true = get_predictions(model, data_loader, device)

    return plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        title,
        save_path,
        figsize,
        normalize,
    )


def main():
    begin_time = time.time()
    cfg = ConfigFineGrained()
    cfg_openood = config.Config(*cfg.config_files)

    os.makedirs(cfg.save_dir, exist_ok=True)

    loader_dict = get_dataloader(cfg_openood)
    test_loader = loader_dict["test"]

    model = ResNet18_32x32(num_classes=cfg.num_classes).to(cfg.device)

    ckpt_path = os.path.join(cfg.save_dir, "best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型权重: {ckpt_path}")

    print(f"Loading model from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")

    # 类别顺序需要与数据集中标签编码一致
    # 对应 rml2016/signallist_split.py 中 modulation_labels 中 label >= 0 的顺序:
    # 'QAM64': 0, 'QPSK': 1, 'BPSK': 2, 'AM-SSB': 3,
    # 'WBFM': 4, 'CPFSK': 5, 'GFSK': 6, 'PAM4': 7
    class_names = [
        "QAM64",
        "QPSK",
        "BPSK",
        "AM-SSB",
        "WBFM",
        "CPFSK",
        "GFSK",
        "PAM4",
    ]
    print(f"Class mapping: {dict(enumerate(class_names))}")

    save_path = os.path.join(
        cfg.save_dir, "confusion_matrix_resnet_fine_grained_normalized.png"
    )
    cm, accuracy, precision, recall, f1 = plot_confusion_matrix_from_model(
        model=model,
        data_loader=test_loader,
        class_names=class_names,
        title="Normalized Confusion Matrix (ResNet18_32x32 Fine-Grained)",
        save_path=save_path,
        figsize=(12, 10),
        normalize=True,
        device=cfg.device,
    )

    print(f"\n用时: {int(time.time() - begin_time)}s")
    return cm, accuracy, precision, recall, f1


if __name__ == "__main__":
    main()


