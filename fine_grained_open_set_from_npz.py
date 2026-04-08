import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# 全局绘图字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False


def compute_threshold_fpr95(conf_id, conf_ood):
    """
    使用 ID / OOD 置信度，按 FPR@95%TPR 方式求阈值：
    - 把 ID 样本作为“正类”，OOD 样本作为“负类”
    - 找到一个阈值，使得 ID 样本中有 95% 的分数 >= 阈值（TPR=95%）
    - 在该阈值下统计 OOD 样本的 FPR
    这里返回的就是这个阈值（你只需要阈值来做开集判别）。
    """
    conf_id = np.asarray(conf_id)
    conf_ood = np.asarray(conf_ood)

    # 对 ID 置信度从高到低排序
    sorted_id = np.sort(conf_id)[::-1]
    # 第 k 个位置使得保留 95% 的 ID（TPR=0.95）
    k = int(np.floor(0.85 * len(sorted_id))) - 1
    k = max(min(k, len(sorted_id) - 1), 0)
    threshold = sorted_id[k]

    # 计算在该阈值下的 FPR（仅用于打印参考）
    fpr = np.mean(conf_ood >= threshold) if len(conf_ood) > 0 else 0.0
    print(f"选取阈值 (TPR≈0.95) = {threshold:.6f}, 对应 OOD FPR ≈ {fpr:.4f}")
    return threshold


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    title="Confusion Matrix",
    save_path=None,
    figsize=(10, 8),
    normalize=False,
):
    # 如果提供了 class_names，则固定混淆矩阵的类别顺序和大小
    if class_names is not None:
        labels = np.arange(len(class_names))
        cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    else:
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

    # 显示矩阵：归一化时显示 0-100，计数时保持原值
    if normalize:
        display_matrix = cm_display * 100.0
        cbar_label = ""
        vmin, vmax = 0, 100
    else:
        display_matrix = cm_display
        cbar_label = "Count"
        vmin, vmax = None, None

    plt.figure(figsize=figsize)
    im = plt.imshow(display_matrix, interpolation="nearest", cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=20, fontweight="bold")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=20)

    tick_marks = np.arange(len(class_names))
    # x轴标签旋转 45°，并整体放大字体
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=20)
    plt.yticks(tick_marks, class_names, fontsize=20)

    thresh = display_matrix.max() / 2.0 if display_matrix.size > 0 else 0.5
    for i, j in itertools.product(range(cm_display.shape[0]), range(cm_display.shape[1])):
        if normalize:
            text = f"{display_matrix[i, j]:.1f}"
        else:
            text = f"{int(display_matrix[i, j])}"
        plt.text(
            j,
            i,
            text,
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm_display[i, j] > thresh else "black",
            fontsize=20,
        )

    plt.ylabel("True Label", fontsize=26)
    plt.xlabel("Predicted Label", fontsize=26)
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


def main():
    base_dir = os.path.join("plot", "fine-grained")
    os.makedirs("results_fine_grained", exist_ok=True)

    # 处理 VOS 和 Ours 两种方法（ID/OOD 分开存储）
    methods = [
        ("Ours", os.path.join(base_dir, "ours-rml201610a.npz"),
         os.path.join(base_dir, "ours-rml201610a_ood.npz")),
        ("VOS", os.path.join(base_dir, "vos-rml201610a.npz"),
         os.path.join(base_dir, "vos-rml201610a_ood.npz")),
    ]

    # 4. 类别名称（8 个已知类 + Unknown）
    known_class_names = [
        "QAM64",
        "QPSK",
        "BPSK",
        "AM-SSB",
        "WBFM",
        "CPFSK",
        "GFSK",
        "PAM4",
    ]

    for method_name, id_npz_path, ood_npz_path in methods:
        if not os.path.exists(id_npz_path) or not os.path.exists(ood_npz_path):
            print(f"[{method_name}] 文件不存在，跳过: {id_npz_path}, {ood_npz_path}")
            continue

        print(f"\n=== 方法: {method_name} ===")

        # 1. 读取 ID 和 OOD 的 npz 结果文件
        id_data = np.load(id_npz_path)
        ood_data = np.load(ood_npz_path)

        pred_id = id_data["pred"]
        conf_id = id_data["conf"]
        label_id = id_data["label"]

        pred_ood = ood_data["pred"]
        conf_ood = ood_data["conf"]
        _ = ood_data["label"]

        # 2. 用 ID / OOD 的置信度计算 FPR@95%TPR 对应的阈值（这里用 90%TPR，你已改过）
        threshold = compute_threshold_fpr95(conf_id=conf_id, conf_ood=conf_ood)

        # 3. 构造带 Unknown 类的标签与预测
        num_known_classes = 8
        unknown_idx = num_known_classes

        y_true_list = []
        y_pred_list = []

        # 3.1 ID 部分：真实标签为 0~7
        for y_true, y_pred, s in zip(label_id, pred_id, conf_id):
            if s < threshold:
                y_pred_open = unknown_idx
            else:
                y_pred_open = int(y_pred)

            y_true_list.append(int(y_true))
            y_pred_list.append(y_pred_open)

        # 3.2 OOD 部分：真实标签统一为 Unknown
        for y_pred, s in zip(pred_ood, conf_ood):
            if s < threshold:
                y_pred_open = unknown_idx
            else:
                y_pred_open = int(y_pred)

            y_true_list.append(unknown_idx)
            y_pred_list.append(y_pred_open)

        y_true = np.array(y_true_list, dtype=int)
        y_pred = np.array(y_pred_list, dtype=int)

        class_names = known_class_names + ["Unknown"]
        print(f"Class mapping (open-set): {dict(enumerate(class_names))}")

        # 5. 绘制带 Unknown 类的归一化混淆矩阵，保存为 PDF
        save_name = f"confusion_matrix_{method_name.lower()}_open_set_from_npz.pdf"
        save_path = os.path.join("results_fine_grained", save_name)
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            title=f"Confusion Matrix ({method_name})",
            save_path=save_path,
            figsize=(12, 10),
            normalize=True,
        )

    # 另外单独处理 SFCR（ID 与 OOD 混在一个 npz，label==-1 视为 OOD）
    sfcr_path = os.path.join(base_dir, "sfcr_rml201610a.npz")
    if os.path.exists(sfcr_path):
        method_name = "SFCR"
        print(f"\n=== 方法: {method_name} ===")

        data = np.load(sfcr_path, allow_pickle=True)
        pred_all = data["pred"]
        conf_all = data["conf"]
        label_all = data["label"]

        id_mask = label_all != -1
        ood_mask = label_all == -1

        pred_id = pred_all[id_mask]
        conf_id = conf_all[id_mask]
        label_id = label_all[id_mask]

        pred_ood = pred_all[ood_mask]
        conf_ood = conf_all[ood_mask]

        if len(conf_id) == 0 or len(conf_ood) == 0:
            print("[SFCR] ID 或 OOD 样本为空，跳过")
        else:
            threshold = compute_threshold_fpr95(conf_id=conf_id, conf_ood=conf_ood)

            num_known_classes = 8
            unknown_idx = num_known_classes

            y_true_list = []
            y_pred_list = []

            # ID 部分
            for y_true, y_pred, s in zip(label_id, pred_id, conf_id):
                if s < threshold:
                    y_pred_open = unknown_idx
                else:
                    y_pred_open = int(y_pred)

                y_true_list.append(int(y_true))
                y_pred_list.append(y_pred_open)

            # OOD 部分
            for y_pred, s in zip(pred_ood, conf_ood):
                if s < threshold:
                    y_pred_open = unknown_idx
                else:
                    y_pred_open = int(y_pred)

                y_true_list.append(unknown_idx)
                y_pred_list.append(y_pred_open)

            y_true = np.array(y_true_list, dtype=int)
            y_pred = np.array(y_pred_list, dtype=int)

            class_names = known_class_names + ["Unknown"]
            print(f"Class mapping (open-set): {dict(enumerate(class_names))}")

            save_name = "confusion_matrix_sfcr_open_set_from_npz.pdf"
            save_path = os.path.join("results_fine_grained", save_name)
            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                title="Confusion Matrix (SFCR)",
                save_path=save_path,
                figsize=(12, 10),
                normalize=True,
            )

    # 同样处理 SR2CNN（ID 与 OOD 混在一个 npz，label==-1 视为 OOD）
    sr2cnn_path = os.path.join(base_dir, "sr2cnn_rml201610a.npz")
    if os.path.exists(sr2cnn_path):
        method_name = "SR2CNN"
        print(f"\n=== 方法: {method_name} ===")

        data = np.load(sr2cnn_path, allow_pickle=True)
        pred_all = data["pred"]
        conf_all = data["conf"]
        label_all = data["label"]

        id_mask = label_all != -1
        ood_mask = label_all == -1

        pred_id = pred_all[id_mask]
        conf_id = conf_all[id_mask]
        label_id = label_all[id_mask]

        pred_ood = pred_all[ood_mask]
        conf_ood = conf_all[ood_mask]

        if len(conf_id) == 0 or len(conf_ood) == 0:
            print("[SR2CNN] ID 或 OOD 样本为空，跳过")
        else:
            threshold = compute_threshold_fpr95(conf_id=conf_id, conf_ood=conf_ood)

            num_known_classes = 8
            unknown_idx = num_known_classes

            y_true_list = []
            y_pred_list = []

            # ID 部分
            for y_true, y_pred, s in zip(label_id, pred_id, conf_id):
                if s < threshold:
                    y_pred_open = unknown_idx
                else:
                    y_pred_open = int(y_pred)

                y_true_list.append(int(y_true))
                y_pred_list.append(y_pred_open)

            # OOD 部分
            for y_pred, s in zip(pred_ood, conf_ood):
                if s < threshold:
                    y_pred_open = unknown_idx
                else:
                    y_pred_open = int(y_pred)

                y_true_list.append(unknown_idx)
                y_pred_list.append(y_pred_open)

            y_true = np.array(y_true_list, dtype=int)
            y_pred = np.array(y_pred_list, dtype=int)

            class_names = known_class_names + ["Unknown"]
            print(f"Class mapping (open-set): {dict(enumerate(class_names))}")

            save_name = "confusion_matrix_sr2cnn_open_set_from_npz.pdf"
            save_path = os.path.join("results_fine_grained", save_name)
            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                title="Confusion Matrix (SR2CNN)",
                save_path=save_path,
                figsize=(12, 10),
                normalize=True,
            )


if __name__ == "__main__":
    main()


