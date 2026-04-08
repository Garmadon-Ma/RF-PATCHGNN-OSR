import os
import copy
from copy import deepcopy
import time
import glob
import random
import pickle
import collections
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from joypy import joyplot
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from sklearn.cluster import KMeans
from scipy.special import ive
from spherecluster import VonMisesFisherMixture
import sklearn.covariance
import openood.utils.comm as comm
from openood.utils import config
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.metrics import compute_all_metrics
from openood.networks.vae import ConditionalVAE, weighted_average

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'  # 数学字体使用 Computer Modern
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设备和随机种子配置
os.environ['PYTHONHASHSEED'] = str(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_loss = 100
best_epoch_idx = 0
best_acc = 0
best_epoch_idx = 0

class Config_EPA:
    def __init__(self):
        # 训练参数
        self.num_classes = 8  # 修改为16个ID类别
        self.feature_dim = 512
        self.epochs = 100       # 增加训练轮次
        self.batch_size = 128
        # 路径参数
        self.save_dir = "results/rml201610a_comix"
        self.device = device
        # 配置文件
        self.config_files = [
            './configs/datasets/rml201610a_comix/rml201610a.yml',
            './configs/networks/resnet18_32x32.yml',
            './configs/pipelines/test/test_ood.yml',
            './configs/preprocessors/base_preprocessor.yml',
            './configs/postprocessors/msp.yml',
        ]
        
class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        effective_classes = num_classes   # 前9个有效类
        P = self.generate_random_orthogonal_matrix(feat_in, effective_classes)
        I = torch.eye(effective_classes)
        one = torch.ones(effective_classes, effective_classes)
        scaling_factor = np.sqrt(effective_classes / (effective_classes - 1))
        self.ori_M = scaling_factor * torch.matmul(P, I - (1.0 / effective_classes) * one).cuda()
        self.ori_M.requires_grad_(False)
        self.LWS = LWS
        self.reg_ETF = reg_ETF

        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        """生成随机正交矩阵"""
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        return P

    def generate_orthogonal_noise(self, M_signal):
        """生成与信号空间正交的噪声向量"""
        u = torch.randn(M_signal.size(0), 1, device=M_signal.device)
        for i in range(M_signal.size(1)):
            u = u - (u.t() @ M_signal[:, i:i+1]) * M_signal[:, i:i+1]
        noise_vec = u / torch.norm(u)
        return noise_vec
    
    def orthogonal_complement(self):
        feat_dim = self.ori_M.shape[0]
        class_dim = self.ori_M.shape[1]
        weight_basis = self.ori_M.T.cpu()  # 转置使每行成为一个基向量
        full_basis = torch.eye(feat_dim)
        complement_basis = []
        for i in range(feat_dim):
            v = full_basis[i]
            v_copy = v.clone()
            for w in weight_basis:
                v_copy = v_copy - torch.dot(v_copy, w) * w / torch.dot(w, w)
            if torch.norm(v_copy) > 1e-6:
                v_copy = v_copy / torch.norm(v_copy)
                complement_basis.append(v_copy)
        if complement_basis:
            orthogonal_complement = torch.stack(complement_basis)
        else:
            orthogonal_complement = torch.zeros((0, feat_dim))      
        return orthogonal_complement
    
    def project_to_complement(self, x):
        orthogonal_comp = self.orthogonal_complement().cuda()
        projection = torch.matmul(torch.matmul(x, orthogonal_comp.t()), orthogonal_comp)
        return projection
    
    def forward(self, x):

        logit = x @ self.ori_M  # 计算logits
        return logit

def load_model(model,checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    return model 

def test_orthogonal_complement():
    """测试正交互补子空间的计算是否正确"""
    # 创建一个小型测试实例
    feat_dim = 512
    num_classes = 16
    classifier = ETF_Classifier(feat_dim, num_classes)
    
    # 获取权重矩阵和计算的正交互补子空间
    weight_matrix = classifier.ori_M
    orthogonal_complement = classifier.orthogonal_complement()
    
    print(f"特征维度: {feat_dim}")
    print(f"类别数量: {num_classes} (有效类别)")  # 修正：移除了-1
    print(f"权重矩阵形状: {weight_matrix.shape}")
    print(f"正交互补子空间形状: {orthogonal_complement.shape}")
    
    # 测试1: 检查正交性
    # 计算权重矩阵的每一列与正交互补子空间的每一行的点积
    # 如果它们正交，所有点积应该接近零
    dot_products = torch.matmul(orthogonal_complement, weight_matrix)
    max_dot_product = torch.max(torch.abs(dot_products)).item()
    print(f"最大点积绝对值: {max_dot_product:.8f} (应接近0)")
    
    # 测试2: 检查维度
    # 计算权重矩阵的秩
    _, S, _ = torch.svd(weight_matrix)
    rank_weights = torch.sum(S > 1e-6).item()
    
    # 正交互补子空间的维度
    dim_complement = orthogonal_complement.shape[0]
    
    print(f"权重矩阵的秩: {rank_weights}")
    print(f"正交互补子空间的维度: {dim_complement}")
    print(f"总和: {rank_weights + dim_complement} (应等于特征维度 {feat_dim})")
    
    # 测试3: 验证投影功能
    # 创建一个随机向量
    random_vector = torch.randn(1, feat_dim)
    
    # 将其投影到正交互补子空间
    projected = classifier.project_to_complement(random_vector)
    
    # 检查投影后的向量与权重矩阵的正交性
    projection_dot_products = torch.matmul(projected, weight_matrix)
    max_proj_dot = torch.max(torch.abs(projection_dot_products)).item()
    print(f"投影向量与权重的最大点积绝对值: {max_proj_dot:.8f} (应接近0)")
    
    return max_dot_product < 1e-5 and (rank_weights + dim_complement == feat_dim) and max_proj_dot < 1e-5

class AdaptiveModulationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18_32x32(num_classes=config_epa.num_classes)
        self.classifier = ETF_Classifier(config_epa.feature_dim, config_epa.num_classes)

    def forward(self, x):
        _, features = self.encoder(x)
        return self.classifier(features), features

class TrainingManager:
    def __init__(self, model):
        self.model = model.to(device)
        self.ce = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.model.encoder.parameters(), 'lr': 1e-3},  # 基础学习率
        ])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100)
    
    def train_phase1(self, train_loader, val_loader):
        global best_acc, best_epoch_idx
        best_acc = 0
        best_epoch_idx = 0

        for epoch in range(config_epa.epochs):
   
            train_metrics = self.train_epoch(train_loader, epoch + 1)
            val_metrics = eval_acc(self.model, val_loader, epoch + 1)
            save_model(self.model,  val_metrics, config_epa.epochs)
            report(train_metrics, val_metrics)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        print(f"Number of batches in epoch {epoch}: {len(loader)}")
        
        for batch in tqdm(loader, desc=f'Epoch {epoch}'):
            x, y = batch['data'].to(device), batch['label'].to(device)
  
            # 分支F的前向传播和反向传播
            self.optimizer.zero_grad()
            logits, features = self.model(x)
            loss = self.ce(logits, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            _, pred = torch.max(logits, dim=1)  # 选择距离最小的索引
            correct += (pred == y).sum().item()
            total += y.size(0)

        metrics = {
            'epoch_idx': epoch,
            'loss': total_loss / len(loader),
            'acc': correct / total
        }
        return metrics

def eval_acc(model, data_loader, epoch_idx):
    data_loaders = data_loader if isinstance(data_loader, list) else [data_loader]

    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        val_data_loader = iter(data_loader)
        for train_step in tqdm(range(1, len(val_data_loader) + 1),
                                desc='Eval: ', position=0, leave=True):
            batch = next(val_data_loader)
            x = batch['data'].to(device)
            y = batch['label'].to(device)



            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            _, pred = torch.max(logits, dim=1)  # 获取最小距离及其对应的索引作为预测
            
            total_correct += (pred == y).sum().item()
            total_samples += y.size(0)
            total_batches += 1
            total_loss += loss.item()
             
    metrics = {}
    metrics['epoch_idx'] = epoch_idx
    metrics['loss'] = save_metrics(total_loss / len(data_loader))
    metrics['acc'] = save_metrics(total_correct / total_samples)
    return metrics

def save_metrics(value):
    all_values = comm.gather(value)
    return sum(all_values)

def save_model(net, val_metrics, num_epochs):
    global best_acc, best_epoch_idx
    output_dir = config_epa.save_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir), exist_ok=True)
    try:
        state_dict = net.module.state_dict()
    except AttributeError:
        state_dict = net.state_dict()


    if val_metrics['acc'] >= best_acc :
        best_acc = val_metrics['acc']
        old_fname = f'best_epoch{best_epoch_idx}_acc{best_acc:.4f}.ckpt'
        old_pth = os.path.join(output_dir, old_fname)
        Path(old_pth).unlink(missing_ok=True)

        best_epoch_idx = val_metrics['epoch_idx']
        best_acc = val_metrics['acc']

        torch.save(state_dict, os.path.join(output_dir,'best.ckpt'))
        save_fname = f'best_epoch{best_epoch_idx}_acc{best_acc:.4f}.ckpt'
        torch.save(state_dict, os.path.join(output_dir, save_fname))

    if val_metrics['epoch_idx'] == num_epochs:
            save_fname = f'last_epoch{val_metrics["epoch_idx"]}_acc{val_metrics["acc"]:.4f}.ckpt'
            torch.save(state_dict, os.path.join(output_dir, save_fname))

def report(train_metrics, val_metrics):
    print('\n  Epoch {:03d} | Time {:5d}s | Train Loss {:.4f} | '
          'Test Loss {:.3f} | Test Acc {:.2f}'.format(
              train_metrics['epoch_idx'],
              int(time.time() - begin_time),
              train_metrics['loss'],
              val_metrics['loss'],
              100.0 * val_metrics['acc'],
            ),
          flush=True)

def get_predictions(model, data_loader):
    """获取模型在数据加载器上的预测结果"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Getting predictions'):
            x = batch['data'].to(device)
            y = batch['label'].to(device)
            
            logits, _ = model(x)
            _, pred = torch.max(logits, dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix', 
                         save_path=None, figsize=(10, 8), normalize=False):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        title: 图表标题
        save_path: 保存路径，如果为None则不保存
        figsize: 图像大小
        normalize: 是否归一化显示百分比
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 3)
    
    # 如果没有提供类别名称，使用数字标签
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 绘制热力图
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    # 设置坐标轴
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 添加数值标注
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12)
    
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()
    
    # 计算并打印准确率
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"总体准确率: {accuracy:.4f}")
    
    # 计算每个类别的精确率、召回率和F1分数
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * precision * recall / (precision + recall)
    
    print("\n各类别性能指标:")
    print(f"{'类别':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    print("-" * 40)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<10} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f}")
    
    return cm, accuracy, precision, recall, f1

def plot_confusion_matrix_from_model(model, data_loader, class_names=None, 
                                   title='Confusion Matrix', save_path=None, 
                                   figsize=(10, 8), normalize=False):
    """
    直接从模型和数据加载器绘制混淆矩阵
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        class_names: 类别名称列表
        title: 图表标题
        save_path: 保存路径
        figsize: 图像大小
        normalize: 是否归一化显示百分比
    """
    print("正在获取预测结果...")
    y_pred, y_true = get_predictions(model, data_loader)
    
    return plot_confusion_matrix(y_true, y_pred, class_names, title, 
                               save_path, figsize, normalize)

if __name__ == '__main__':

    begin_time = time.time()
    config_epa = Config_EPA()
    configopenood = config.Config(*config_epa.config_files)
    os.makedirs(config_epa.save_dir, exist_ok=True)
    loader_dict = get_dataloader(configopenood)

    train_loader = loader_dict['train']
    val_loader = loader_dict['val']
    test_loader = loader_dict['test']
    model = AdaptiveModulationModel()


    #训练
    trainer = TrainingManager(model)
    trainer.train_phase1(train_loader, val_loader)

    #混淆矩阵
    # model = load_model(model, 'results/rml201610a_comix/best.ckpt').to(device)
    # # 根据标签映射定义类别名称
    # class_names = ['QAM16', 'QAM64', 'PSK', 'QPSK', 'BPSK']
    # plot_confusion_matrix_from_model(
    #     model=model,
    #     data_loader=test_loader,
    #     class_names=class_names,
    #     title='Normalized Confusion Matrix',
    #     save_path=os.path.join(config_epa.save_dir, 'confusion_matrix_normalized.png'),
    #     figsize=(12, 10),
    #     normalize=True
    # )