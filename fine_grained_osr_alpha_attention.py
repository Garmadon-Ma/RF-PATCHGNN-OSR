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
from scipy import stats
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple
import math
from torch.utils.data import Dataset, DataLoader, Subset
from joypy import joyplot
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from sklearn.cluster import KMeans
from scipy.special import ive
# from spherecluster import VonMisesFisherMixture
import sklearn.covariance

from openood.networks import resnet18_32x32
import openood.utils.comm as comm
from openood.utils import config
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.metrics import compute_all_metrics
from openood.networks.vae import ConditionalVAE, weighted_average

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'  
plt.rcParams['axes.unicode_minus'] = False  

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

best_acc = 0
best_epoch_idx = 0

class Config_gnn:
    def __init__(self):

        self.num_classes = 8  
        self.feature_dim = 128
        self.epochs = 100      
        self.batch_size = 128
        
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.momentum_opt = 0.9
        
        self.save_dir = "results/fctf_rml201610a"
        self.device = device

        self.config_files = [
            './configs/datasets/rml201610a_comix/rml201610a.yml',
            './configs/networks/resnet18_32x32.yml',
            './configs/pipelines/test/test_ood.yml',
            './configs/preprocessors/base_preprocessor.yml',
            './configs/postprocessors/msp.yml',
        ]
        
        # FCTF-Net 
        self.signal_length = 128      # 信号长度
        self.patch_length = 8       # 每个patch的长度
        self.patch_stride = 4        # patch的步长
        self.local_window = 4         # 图连接的局部窗口大小
        self.gnn_hidden_dim = 128      # GNN隐藏层维度
        self.gnn_layers = 4           # GNN层数
        self.rnn_hidden_dim = 128     # RNN隐藏层维度
        self.rnn_layers = 4           # RNN层数
        self.rnn_type = 'GRU'        # 'LSTM' 或 'GRU'
        self.dropout = 0.1             # Dropout比率
        
class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(ETF_Classifier, self).__init__()
        effective_classes = num_classes   
        P = self.generate_random_orthogonal_matrix(feat_in, effective_classes)
        I = torch.eye(effective_classes)
        one = torch.ones(effective_classes, effective_classes)
        scaling_factor = np.sqrt(effective_classes / (effective_classes - 1))

        self.ori_M = scaling_factor * torch.matmul(P, I - (1.0 / effective_classes) * one).cuda()
        self.ori_M.requires_grad_(False)

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        return P
    
    def orthogonal_complement(self):
        """使用施密特正交化计算权重矩阵的正交互补子空间"""
        # 获取权重矩阵的形状
        feat_dim = self.ori_M.shape[0]
        class_dim = self.ori_M.shape[1]
        
        # 首先获取权重矩阵的列向量作为初始子空间的基
        weight_basis = self.ori_M.T.cpu()  # 转置使每行成为一个基向量
        
        # 创建一组完整的基向量（包括单位向量）
        full_basis = torch.eye(feat_dim)
        
        # 使用施密特正交化过程找到正交互补子空间
        complement_basis = []
        
        for i in range(feat_dim):
            v = full_basis[i]
            
            # 检查v是否与权重子空间线性无关
            # 通过计算v与权重基向量的投影，然后检查剩余部分是否显著
            v_copy = v.clone()
            
            # 从v中减去它在权重基向量上的投影
            for w in weight_basis:
                v_copy = v_copy - torch.dot(v_copy, w) * w / torch.dot(w, w)
            
            # 如果剩余向量的范数足够大，则它是互补子空间的一部分
            if torch.norm(v_copy) > 1e-6:
                # 归一化
                v_copy = v_copy / torch.norm(v_copy)
                complement_basis.append(v_copy)
        
        # 将互补基向量堆叠成矩阵
        if complement_basis:
            orthogonal_complement = torch.stack(complement_basis)
        else:
            # 如果没有找到互补基向量，返回空矩阵
            orthogonal_complement = torch.zeros((0, feat_dim))
        
        return orthogonal_complement
    
    def project_to_complement(self, x):
        orthogonal_comp = self.orthogonal_complement().to(x.device)
        # 计算投影
        projection = torch.matmul(torch.matmul(x, orthogonal_comp.t()), orthogonal_comp)
        return projection
    
    def forward(self, x):

        logit = x @ self.ori_M 
        return logit

class FCTF(nn.Module):
    #信号->patch图
    def __init__(self, patch_length: int = 32, patch_stride: int = 32, 
                 local_window: int = 8, signal_length: int = 128):  
        super(FCTF, self).__init__()
        
        self.patch_length = patch_length
        self.patch_stride = patch_stride

        #图每个节点的有边上限数
        self.local_window = local_window 
        self.signal_length = signal_length
        
        #patch数量
        self.num_patches = ((signal_length - patch_length) // patch_stride) + 1

        #可训练的边权重
        self.edge_weights = nn.Parameter(torch.randn(local_window))
        
        # 初始化掩码矩阵 
        self.register_buffer('adj_mask', self._create_adjacency_mask())
        
        # 预计算边索引和距离
        self.register_buffer('edge_index', self._precompute_edge_index())
        self.register_buffer('edge_distance', self._precompute_edge_distance())
    
    #排除自己 连接前local_window个点和后local_window个点 局部
    def _create_adjacency_mask(self) -> torch.Tensor:
        # 向量化实现，避免循环
        i_indices = torch.arange(self.patch_length, dtype=torch.long).unsqueeze(1)
        j_indices = torch.arange(self.patch_length, dtype=torch.long).unsqueeze(0)
        distance = torch.abs(i_indices - j_indices)
        mask = ((distance > 0) & (distance <= self.local_window)).float()
        return mask
    
    def _precompute_edge_index(self) -> torch.Tensor:
        #无向图
        edge_list = []
        for i in range(self.patch_length):
            for j in range(self.patch_length):
                if 0 < abs(i - j) <= self.local_window:
                    edge_list.append([i, j])
        # 有向图
        # for i in range(self.patch_length):
        #     for j in range(i+1, min(i+self.local_window+1, self.patch_length)):
        #         edge_list.append([i, j])  # 只连接前向
        #         if len(edge_list) == 0:

        if len(edge_list) == 0:
            edge_list = [[i, i] for i in range(self.patch_length)]
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    def _precompute_edge_distance(self) -> torch.Tensor:
        distances = []
        for i in range(self.patch_length):
            for j in range(self.patch_length):
                if 0 < abs(i - j) <= self.local_window:
                    distances.append(abs(i - j))
        if len(distances) == 0:
            distances = [1] * self.patch_length  # 自环距离设为1
        return torch.tensor(distances, dtype=torch.long) - 1  # 减1用于索引edge_weights


    ''' 有边
        adj_mask = [
        [0, 1, 1, 1, 0, 0, 0, 0],  # i=0: 连接节点1,2,3
        [1, 0, 1, 1, 1, 0, 0, 0],  # i=1: 连接节点0,2,3,4
        [1, 1, 0, 1, 1, 1, 0, 0],  # i=2: 连接节点0,1,3,4,5
        [1, 1, 1, 0, 1, 1, 1, 0],  # i=3: 连接节点0,1,2,4,5,6
        [0, 1, 1, 1, 0, 1, 1, 1],  # i=4: 连接节点1,2,3,5,6,7
        [0, 0, 1, 1, 1, 0, 1, 1],  # i=5: 连接节点2,3,4,6,7
        [0, 0, 0, 1, 1, 1, 0, 1],  # i=6: 连接节点3,4,5,7
        [0, 0, 0, 0, 1, 1, 1, 0],  # i=7: 连接节点4,5,6
    ]
    '''
    '''
        edge_index = [
            [0,1], [0,2], [0,3],           # 节点0的边
            [1,0], [1,2], [1,3], [1,4],    # 节点1的边
            [2,0], [2,1], [2,3], [2,4], [2,5],  # 节点2的边
            [3,0], [3,1], [3,2], [3,4], [3,5], [3,6],  # 节点3的边
            [4,1], [4,2], [4,3], [4,5], [4,6], [4,7],  # 节点4的边
            [5,2], [5,3], [5,4], [5,6], [5,7],  # 节点5的边
            [6,3], [6,4], [6,5], [6,7],   # 节点6的边
            [7,4], [7,5], [7,6],          # 节点7的边
    ]
        edge_attr = [
            0.5, 0.3, 0.1,              # 节点0的边权重
            0.5, 0.5, 0.3, 0.1,         # 节点1的边权重
            0.5, 0.5, 0.3, 0.3, 0.1,    # 节点2的边权重
            ...                          # 继续
    ]
    '''

    def _construct_patch_graph(self, patch_data: torch.Tensor) -> Data:
        node_features = patch_data 
        edge_index = self.edge_index
        edge_attr = self.edge_weights[self.edge_distance]
            
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    def forward(self, iq_signal: torch.Tensor) -> Batch:
        """
        优化的批量图构建，避免Python循环
        返回Batch对象而不是List[Data]
        """
        if iq_signal.dim() == 3 and iq_signal.shape[1] == 2:
            iq_signal = iq_signal.transpose(1, 2)  # [B, 2, L] -> [B, L, 2]
        
        batch_size = iq_signal.shape[0]
        device = iq_signal.device
        
        # 向量化提取所有patches: 
        # 将 [B, L, 2] 转换为 [B, 2, L] 以便使用unfold
        iq_signal_t = iq_signal.transpose(1, 2)  # [B, L, 2] -> [B, 2, L]
        
        # 对每个通道分别unfold，然后合并
        patches_list = []
        for c in range(2):
            signal_2d = iq_signal_t[:, c:c+1, :].unsqueeze(2)  # [B, 1, 1, L]
            unfolded = F.unfold(
                signal_2d,
                kernel_size=(1, self.patch_length),
                stride=(1, self.patch_stride)
            )  # [B, patch_length, num_patches]
            # 转置: [B, num_patches, patch_length]
            patches_list.append(unfolded.transpose(1, 2))
        
        # [B, num_patches, patch_length, 2]
        patches = torch.stack(patches_list, dim=-1)
        
        # [B * num_patches, patch_length, 2]
        # [batch0_patch0, batch0_patch1, ..., batch1_patch0, ...]
        patches_flat = patches.reshape(batch_size * self.num_patches, self.patch_length, 2)
        
        # [B * num_patches * patch_length, 2]
        node_features = patches_flat.reshape(-1, 2)
        
        # 为每个图创建边索引
        num_edges = self.edge_index.shape[1]
        num_graphs = batch_size * self.num_patches
        num_nodes_per_graph = self.patch_length
        
        # 向量化创建所有图的边索引偏移
        graph_offsets = torch.arange(0, num_graphs * num_nodes_per_graph, 
                                    num_nodes_per_graph, device=device)
        # [num_graphs, 1] 
        offsets = graph_offsets.view(-1, 1)
        # [1, num_edges] 
        edge_index_base = self.edge_index.to(device)  # [2, num_edges]
        # [num_graphs, 2, num_edges]
        edge_index_expanded = edge_index_base.unsqueeze(0).expand(num_graphs, -1, -1)
        # 添加偏移: 
        batch_edge_index_per_graph = edge_index_expanded + offsets.unsqueeze(1)
        # 重塑为[2, num_edges * num_graphs]
        batch_edge_index = batch_edge_index_per_graph.permute(1, 0, 2).reshape(2, -1)
        
        # 为每个图创建边属性（所有图的边属性相同）
        edge_attr = self.edge_weights[self.edge_distance]  # [num_edges]
        batch_edge_attr = edge_attr.repeat(num_graphs)  # [num_edges * num_graphs]
        
        # 创建batch向量: [0,0,...,0, 1,1,...,1, ...] 每个图有patch_length个节点
        batch = torch.arange(num_graphs, device=device).repeat_interleave(num_nodes_per_graph)
        
        # 直接创建Batch对象，避免创建大量Data对象
        batch_graph = Batch(
            x=node_features,  # [B * num_patches * patch_length, 2]
            edge_index=batch_edge_index,
            edge_attr=batch_edge_attr,
            batch=batch
        )
        
        return batch_graph

class PatchGNN(nn.Module):#目前还是patch级别``
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, 
                 num_layers: int = 3, dropout: float = 0.1):
        super(PatchGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(
            GraphSAGE(input_dim, hidden_dim, num_layers=1)#图采样和图聚合 
        )
        for _ in range(1, num_layers):
            self.gnn_layers.append(
                GraphSAGE(hidden_dim, hidden_dim, num_layers=1)
            )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, batch_graph: Batch) -> torch.Tensor:
        '''
        edge_index = [
        [0, 0, 0, 1, 1, 1, ..., 31, 31, 31, 32, 32, ...],  # 源节点
        [1, 2, 3, 0, 2, 3, ..., 30, 31, 0, 33, 34, ...]   # 目标节点
        ]
        batch = [
        0, 0, 0, ..., 0,        # 前32个节点属于图0
        1, 1, 1, ..., 1,        # 接下来32个节点属于图1
        2, 2, 2, ..., 2,        # 接下来32个节点属于图2
        ...
        15, 15, 15, ..., 15    # 最后32个节点属于图15
        ]
        '''
        x, edge_index, batch = batch_graph.x, batch_graph.edge_index, batch_graph.batch
        
        # 如果有边属性，将其作为边权重
        edge_weight = getattr(batch_graph, 'edge_attr', None)#边权重
        
        # 通过GNN层
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_new = gnn_layer(x, edge_index, edge_weight=edge_weight)
            x_new = self.layer_norms[i](x_new)
            x_new = self.activation(x_new)
            x_new = self.dropout(x_new)
            
            # 残差连接
            if i > 0 and x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        
        #平均池化
        graph_features = global_mean_pool(x, batch)
        
        # # 输出投影
        # graph_features = self.output_proj(graph_features)
        # graph_features = self.activation(graph_features)
        
        return graph_features

class TemporalRNN(nn.Module):#patch序列->时序特征   横向
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, 
                 num_layers: int = 2, rnn_type: str = 'LSTM', dropout: float = 0.1):
        super(TemporalRNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # 层归一化
        # self.layer_norm = nn.LayerNorm(hidden_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, patch_features: torch.Tensor, num_patches: int) -> torch.Tensor:
        total_features = patch_features.shape[0]
        batch_size = total_features // num_patches
        
        # reshape为序列形式: [batch_size, num_patches, feature_dim]

        sequence_features = patch_features.view(batch_size, num_patches, -1)
        
        if self.rnn_type == 'LSTM':
            output, (hidden, _) = self.rnn(sequence_features)
        else: 
            output, hidden = self.rnn(sequence_features)
    

        sequence_representation = output[:, -1, :]  # [batch_size, hidden_dim]
        
        # sequence_representation = self.layer_norm(sequence_representation)
        # sequence_representation = self.dropout(sequence_representation)
        
        return sequence_representation

class AttentionFusion(nn.Module):
    """注意力融合模块，用于融合两个相同维度的特征"""
    def __init__(self, feature_dim: int = 128, dropout: float = 0.1):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        
        # 注意力权重计算（用于决定两个特征的融合比例）
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        融合两个特征
        Args:
            feat1: [B, feature_dim] - temporal_features
            feat2: [B, feature_dim] - combined_features
        Returns:
            fused_feature: [B, feature_dim]
        """
        # 计算注意力权重来决定两个特征的融合比例
        concat_feat = torch.cat([feat1, feat2], dim=-1)  # [B, 2*feature_dim]
        fusion_weights = self.attention_weights(concat_feat)  # [B, 2]
        
        # 加权融合
        fused = fusion_weights[:, 0:1] * feat1 + fusion_weights[:, 1:2] * feat2  # [B, feature_dim]
        
        return fused

class FCTFNet(nn.Module):
    def __init__(self, num_classes: int = 8, signal_length: int = 128,
                 patch_length: int = 32, patch_stride: int = 32,
                 local_window: int = 8, gnn_hidden_dim: int = 64,
                 gnn_layers: int = 3, rnn_hidden_dim: int = 128,
                 rnn_layers: int = 2, rnn_type: str = 'LSTM',
                 dropout: float = 0.1):
        super(FCTFNet, self).__init__()
        
        self.signal_length = signal_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride

        self.fctf = FCTF(
            patch_length=patch_length,
            patch_stride=patch_stride,
            local_window=local_window,
            signal_length=signal_length
        )
        
        self.num_patches = self.fctf.num_patches


        self.fusion_weights = nn.Parameter(torch.zeros(num_classes))

        self.encoder = ResNet18_32x32()

        self.patch_gnn = PatchGNN(
            input_dim=2, 
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout
        )
        self.temporal_rnn = TemporalRNN(
            input_dim=gnn_hidden_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_layers,
            rnn_type=rnn_type,
            dropout=dropout
        )
        # 
        self.attention_fusion = AttentionFusion(
            feature_dim=rnn_hidden_dim,
            dropout=dropout
        )
        #etf
        self.etf_classifier = ETF_Classifier(feat_in=rnn_hidden_dim, num_classes=num_classes)
        
       
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        iq_signal = data.squeeze(1)[:, :2] 
        batch_graph = self.fctf(iq_signal).to(device)
        patch_features = self.patch_gnn(batch_graph)          
        temporal_features = self.temporal_rnn(patch_features, self.num_patches)

        combined_signal = data[ :, :, 2:]
        _, combined_features = self.encoder(combined_signal)
        # 使用注意力机制融合两个特征
        features = self.attention_fusion(temporal_features, combined_features)
        logits = self.etf_classifier(features)

        return logits

class FCTFTrainer(nn.Module):
    def __init__(self, net: nn.Module, train_loader: DataLoader, config_gnn) -> None:
        super(FCTFTrainer, self).__init__()
        self.net = net
        self.train_loader = train_loader
        self.config_gnn = config_gnn

        self.optimizer = torch.optim.Adam(
            net.parameters(),
            lr=config_gnn.lr,
            weight_decay=config_gnn.weight_decay
        )
        
        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * \
                        (1 + np.cos(step / total_steps * np.pi))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config_gnn.epochs * len(train_loader),
                1,
                1e-6 / config_gnn.lr,
            ),
        )
        
        self.ce = nn.CrossEntropyLoss()

    def train_epoch(self, epoch_idx):
        self.net.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(self.train_loader, desc=f'Train Epoch {epoch_idx}'):
            data = batch['data'].to(self.config_gnn.device)
            target = batch['label'].to(self.config_gnn.device)
            

            self.optimizer.zero_grad()
            logits = self.net(data)
            loss = self.ce(logits, target)
            loss.backward() 
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
        
        metrics = {
            'epoch_idx': epoch_idx,
            'loss': total_loss / len(self.train_loader),
            'acc': correct / total
        }
        return metrics
    
class FCTFTrainingManager:
    def __init__(self, config_gnn):
        self.config_gnn = config_gnn
        self.device = config_gnn.device
        self.train_losses = []
        self.val_accuracies = []
        self.ce = nn.CrossEntropyLoss()
        
    def train(self, train_loader, val_loader):
        model = FCTFNet(
            num_classes=self.config_gnn.num_classes,
            signal_length=self.config_gnn.signal_length,
            patch_length=self.config_gnn.patch_length,
            patch_stride=self.config_gnn.patch_stride,
            local_window=self.config_gnn.local_window,
            gnn_hidden_dim=self.config_gnn.gnn_hidden_dim,
            gnn_layers=self.config_gnn.gnn_layers,
            rnn_hidden_dim=self.config_gnn.rnn_hidden_dim,
            rnn_layers=self.config_gnn.rnn_layers,
            rnn_type=self.config_gnn.rnn_type,
            dropout=self.config_gnn.dropout
        ).to(self.config_gnn.device)
        
        trainer = FCTFTrainer(model, train_loader, self.config_gnn)
        print("Starting FCTF-Net training...")        
        for epoch in range(1, self.config_gnn.epochs + 1):

            train_metrics = trainer.train_epoch(epoch)
            val_metrics = self.validate(model, val_loader, epoch)         
            self.save_model(model, val_metrics, self.config_gnn.epochs)         
            self.report(train_metrics, val_metrics)      
        return model, trainer
    
    def validate(self, model, val_loader, epoch_idx):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Val Epoch {epoch_idx}'):
                data = batch['data'].to(self.device)
                target = batch['label'].to(self.device)
                
                logits = model(data)
                loss = F.cross_entropy(logits, target)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(target).sum().item()
                total += target.size(0)  
        
        metrics = {
            'epoch_idx': epoch_idx,
            'loss': val_loss / len(val_loader),
            'acc': correct / total
        }
        return metrics
    
    def save_model(self, net, val_metrics, num_epochs):
        global best_acc, best_epoch_idx
        output_dir = self.config_gnn.save_dir
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            state_dict = net.module.state_dict()
        except AttributeError:
            state_dict = net.state_dict()
        
        if val_metrics['acc'] >= best_acc:
            old_fname = f'best_epoch{best_epoch_idx}_acc{best_acc:.4f}.ckpt'
            old_pth = os.path.join(output_dir, old_fname)
            Path(old_pth).unlink(missing_ok=True)
            
            best_epoch_idx = val_metrics['epoch_idx']
            best_acc = val_metrics['acc']
            
            torch.save(state_dict, os.path.join(output_dir, 'best.ckpt'))
            save_fname = f'best_epoch{best_epoch_idx}_acc{best_acc:.4f}.ckpt'
            torch.save(state_dict, os.path.join(output_dir, save_fname))
            print(f"Best model saved: {save_fname}")
        
        if val_metrics['epoch_idx'] == num_epochs:
            save_fname = f'last_epoch{val_metrics["epoch_idx"]}_acc{val_metrics["acc"]:.4f}.ckpt'
            torch.save(state_dict, os.path.join(output_dir, save_fname))
            print(f"Final model saved: {save_fname}")
    
    def report(self, train_metrics, val_metrics):
        """Training progress report"""
        print('\n  Epoch {:03d} | Train Loss {:.4f} | '
              'Val Loss {:.3f} | Val Acc {:.2f}%'.format(
                  train_metrics['epoch_idx'],
                  train_metrics['loss'],
                  val_metrics['loss'],
                  100.0 * val_metrics['acc'],
              ),
              flush=True)

def get_predictions(model, data_loader, device):
    """获取模型在数据加载器上的预测结果"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Getting predictions'):
            data = batch['data'].to(device).squeeze(1)[:, :2]  # 只取前两行（I-Q复合信号）
            labels = batch['label'].to(device)
            
            logits = model(data)
            _, pred = torch.max(logits, dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
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
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
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
    
    # 处理除零情况
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    
    print("\n各类别性能指标:")
    print(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    print("-" * 45)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f}")
    
    return cm, accuracy, precision, recall, f1

def plot_confusion_matrix_from_model(model, data_loader, class_names=None, 
                                   title='Confusion Matrix', save_path=None, 
                                   figsize=(10, 8), normalize=False, device=None):
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
        device: 设备
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("正在获取预测结果...")
    y_pred, y_true = get_predictions(model, data_loader, device)
    
    return plot_confusion_matrix(y_true, y_pred, class_names, title, 
                               save_path, figsize, normalize)

def test_and_plot_confusion_matrix(checkpoint_path=None, class_names=None, normalize=True):
    """
    加载训练好的模型并在测试集上绘制混淆矩阵
    
    Args:
        checkpoint_path: 模型检查点路径，如果为None则使用默认路径
        class_names: 类别名称列表，如果为None则使用默认名称
        normalize: 是否归一化显示百分比
    """
    config_gnn = Config_gnn()
    configopenood = config.Config(*config_gnn.config_files)
    
    # 获取测试数据加载器
    loader_dict = get_dataloader(configopenood)
    test_loader = loader_dict['test']
    
    # 创建模型
    model = FCTFNet(
        num_classes=config_gnn.num_classes,
        signal_length=config_gnn.signal_length,
        patch_length=config_gnn.patch_length,
        patch_stride=config_gnn.patch_stride,
        local_window=config_gnn.local_window,
        gnn_hidden_dim=config_gnn.gnn_hidden_dim,
        gnn_layers=config_gnn.gnn_layers,
        rnn_hidden_dim=config_gnn.rnn_hidden_dim,
        rnn_layers=config_gnn.rnn_layers,
        rnn_type=config_gnn.rnn_type,
        dropout=config_gnn.dropout
    ).to(config_gnn.device)
    
    # 加载训练好的模型
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config_gnn.save_dir, 'best.ckpt')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=config_gnn.device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully!")
    else:
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Available files in save directory:")
        if os.path.exists(config_gnn.save_dir):
            for file in os.listdir(config_gnn.save_dir):
                print(f"  - {file}")
        return None
    
    model = model.to(config_gnn.device)
    model.eval()
    
    # 设置类别名称
    if class_names is None:
        class_names = ['QAM16', 'QAM64', '8PSK', 'QPSK', 'BPSK', 'AM-DSB', 'AM-SSB', 'WBFM']
    
    print(f"Class mapping: {dict(enumerate(class_names))}")
    
    # 绘制混淆矩阵
    save_path = os.path.join(config_gnn.save_dir, 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png')
    cm, accuracy, precision, recall, f1 = plot_confusion_matrix_from_model(
        model=model,
        data_loader=test_loader,
        class_names=class_names,
        title='Normalized Confusion Matrix' if normalize else 'Confusion Matrix',
        save_path=save_path,
        figsize=(12, 10),
        normalize=normalize,
        device=config_gnn.device
    )
    
    return cm, accuracy, precision, recall, f1

def load_model(model,checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    return model 

if __name__ == '__main__':
    config_gnn = Config_gnn()
    configopenood = config.Config(*config_gnn.config_files)
    os.makedirs(config_gnn.save_dir, exist_ok=True)
    
    loader_dict = get_dataloader(configopenood)
    train_loader = loader_dict['train']
    val_loader = loader_dict['val']
    test_loader = loader_dict['test']
    
    #train
    training_manager = FCTFTrainingManager(config_gnn)
    trained_model, trainer = training_manager.train(train_loader, val_loader)
    


    #confusion_matrix
    # cm, accuracy, precision, recall, f1 = test_and_plot_confusion_matrix(
    #     checkpoint_path=None,  
    #     class_names=None,  
    #     normalize=True  
    # )

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Best epoch: {best_epoch_idx}")
    print(f"Model saved to: {config_gnn.save_dir}")