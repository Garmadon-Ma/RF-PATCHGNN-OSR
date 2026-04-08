import os
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from openood.networks.resnet18_32x32 import ResNet18_32x32
from gnn import DTSGNet, Config as GNNConfig


class DPLNet(nn.Module):
    """DPL 上分支：ResNet18_32x32 + 两层 MLP 投影到 128 维并 L2 归一化。"""

    def __init__(self, num_classes: int = 26, feat_dim: int = 128):
        super(DPLNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = ResNet18_32x32(num_classes=num_classes)
        self.feat_dim = feat_dim

        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.backbone(x)
        projected = self.projector(features)
        normalized = F.normalize(projected, dim=-1)
        return normalized


def initialize_etf_prototypes(num_classes: int, feat_dim: int) -> torch.Tensor:
    a = np.random.random(size=(feat_dim, num_classes))
    P, _ = np.linalg.qr(a)
    P = torch.tensor(P).float()
    I_K = torch.eye(num_classes)
    ones_K = torch.ones(num_classes, num_classes)
    scaling = math.sqrt(num_classes / (num_classes - 1))
    W = scaling * torch.matmul(P, I_K - (1.0 / num_classes) * ones_K)
    return W  # [D, C]


class PrototypeBank(nn.Module):
    """每个类别的动态细粒度原型库（DP-means 风格在线增加）。

    - 在线更新：对每个样本，若其到最近原型的距离 > lambda，则创建新原型（上限 max_prototypes_per_class）
    - 原型更新：指数移动平均
    - 原型合并：当同类原型之间距离小于 merge_threshold 时合并
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        max_prototypes_per_class: int = 5,
        dp_lambda: float = 0.7,
        ema_momentum: float = 0.9,
        merge_threshold: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.max_prototypes_per_class = max_prototypes_per_class
        self.dp_lambda = dp_lambda
        self.ema_momentum = ema_momentum
        self.merge_threshold = merge_threshold

        # 使用 Python 列表存储每类原型张量列表与计数，便于动态增删
        self.register_buffer('device_anchor', torch.zeros(1))
        self.prototypes: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]
        self.counts: List[List[int]] = [[] for _ in range(num_classes)]

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device_anchor.device)

    def get_class_prototypes(self, c: int) -> Optional[torch.Tensor]:
        if len(self.prototypes[c]) == 0:
            return None
        return torch.stack(self.prototypes[c], dim=0)  # [Kc, D]

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算到同类与异类原型的最邻近相似度。
        返回: pos_sim [B], neg_sim [B]（基于余弦相似度）。
        """
        B = z.shape[0]
        device = z.device
        pos_sim = torch.full((B,), -1.0, device=device)
        neg_sim = torch.full((B,), -1.0, device=device)

        z_n = F.normalize(z, dim=-1)
        for c in range(self.num_classes):
            class_mask = (y == c)
            if not torch.any(class_mask):
                continue
            z_c = z_n[class_mask]  # [Bc, D]
            protos = self.get_class_prototypes(c)
            if protos is None:
                continue
            protos_n = F.normalize(protos, dim=-1)  # [Kc, D]
            sim_mat = torch.matmul(z_c, protos_n.T)  # [Bc, Kc]
            pos_sim[class_mask] = sim_mat.max(dim=1).values

        # 异类原型相似度：对所有非 y 的类的原型取最大
        all_protos = []
        all_labels = []
        for c in range(self.num_classes):
            if len(self.prototypes[c]) == 0:
                continue
            p = self.get_class_prototypes(c)
            all_protos.append(p)
            all_labels.extend([c] * p.shape[0])
        if len(all_protos) > 0:
            all_protos = torch.cat(all_protos, dim=0)  # [Kall, D]
            all_protos = F.normalize(all_protos, dim=-1)
            sim_all = torch.matmul(z_n, all_protos.T)  # [B, Kall]
            # mask 自身类别的原型
            proto_labels = torch.tensor(all_labels, device=device)
            same_mask = proto_labels.unsqueeze(0).expand(B, -1).eq(y.unsqueeze(1))
            sim_all_masked = sim_all.masked_fill(same_mask, float('-inf'))
            neg_sim = sim_all_masked.max(dim=1).values

        return pos_sim, neg_sim

    @torch.no_grad()
    def update_online(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """DP-means 风格在线更新与自增长。z 已经 L2 归一化。"""
        z = F.normalize(z, dim=-1)
        device = self.device_anchor.device
        for i in range(z.shape[0]):
            zi = z[i].detach().to(device)
            ci = int(y[i].item())
            # 若无原型，直接创建
            if len(self.prototypes[ci]) == 0:
                self.prototypes[ci].append(zi.clone())
                self.counts[ci].append(1)
                continue
            # 找最近原型（用 1 - 余弦相似度作为距离）
            P = self.get_class_prototypes(ci).to(device)  # [K, D]
            sim = torch.matmul(F.normalize(P, dim=-1), zi)  # [K]
            best_idx = int(torch.argmax(sim).item())
            dist = 1.0 - float(sim[best_idx].item())
            if dist > self.dp_lambda and len(self.prototypes[ci]) < self.max_prototypes_per_class:
                # 新建原型
                self.prototypes[ci].append(zi.clone())
                self.counts[ci].append(1)
            else:
                # EMA 更新
                m = self.ema_momentum
                self.prototypes[ci][best_idx] = F.normalize(
                    m * self.prototypes[ci][best_idx] + (1 - m) * zi,
                    dim=-1,
                )
                self.counts[ci][best_idx] += 1

        # 合并过近的原型
        self._merge_close_prototypes()

    @torch.no_grad()
    def _merge_close_prototypes(self) -> None:
        device = self.device_anchor.device
        for c in range(self.num_classes):
            if len(self.prototypes[c]) <= 1:
                continue
            P = torch.stack(self.prototypes[c], dim=0).to(device)  # [K, D]
            P = F.normalize(P, dim=-1)
            sim = torch.matmul(P, P.T)  # [K, K]
            K = sim.shape[0]
            merged = [False] * K
            new_list: List[torch.Tensor] = []
            new_counts: List[int] = []
            for i in range(K):
                if merged[i]:
                    continue
                group = [i]
                for j in range(i + 1, K):
                    if merged[j]:
                        continue
                    # 1 - sim < merge_threshold => 合并
                    if 1.0 - float(sim[i, j].item()) < self.merge_threshold:
                        merged[j] = True
                        group.append(j)
                # 组内加权平均
                vecs = [self.prototypes[c][idx].to(device) for idx in group]
                cnts = [self.counts[c][idx] for idx in group]
                wsum = sum(cnts)
                centroid = F.normalize(
                    torch.stack(vecs, dim=0).mul(torch.tensor(cnts, device=device).view(-1, 1)).sum(dim=0) / wsum,
                    dim=-1,
                )
                new_list.append(centroid)
                new_counts.append(wsum)
            self.prototypes[c] = new_list
            self.counts[c] = new_counts


class DualBranchNet(nn.Module):
    """双分支网络：DPL 负责类别对齐（粗粒度），GNN 负责类内细粒度原型。

    - 输出三路 logits（dpl/gnn/fuse）与两路特征 z_dpl/z_gnn
    - 共享 ETF 类级原型作为对齐分类器
    - GNN 特征通过 PrototypeBank 做细粒度建模
    """

    def __init__(
        self,
        num_classes: int,
        dpl_feat_dim: int,
        gnn_config: GNNConfig,
        temperature: float = 0.5,
        proto_bank: Optional[PrototypeBank] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = dpl_feat_dim
        self.temperature = temperature

        self.dpl = DPLNet(num_classes=num_classes, feat_dim=dpl_feat_dim)
        self.gnn = DTSGNet(gnn_config)

        rnn_dim = gnn_config.rnn_hidden_dim * (2 if gnn_config.rnn_bidirectional else 1)
        self.proj_gnn = nn.Sequential(
            nn.Linear(rnn_dim, dpl_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(dpl_feat_dim, dpl_feat_dim),
        )

        etf = initialize_etf_prototypes(num_classes, dpl_feat_dim)
        self.register_buffer('etf', etf)  # [D, C]

        self.fuse_head = nn.Sequential(
            nn.Linear(dpl_feat_dim * 2, dpl_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(dpl_feat_dim, num_classes),
        )

        self.proto_bank = proto_bank or PrototypeBank(num_classes, dpl_feat_dim)

    def class_logits(self, z: torch.Tensor) -> torch.Tensor:
        return torch.matmul(z, self.etf) / self.temperature

    def forward(self, img: torch.Tensor, I: torch.Tensor, Q: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 上分支
        z_dpl = self.dpl(img)  # [B, D]
        # 下分支
        gnn_hidden = self.gnn.forward_features(I, Q)  # [B, Dr]
        z_gnn = F.normalize(self.proj_gnn(gnn_hidden), dim=-1)  # [B, D]

        logits_dpl = self.class_logits(z_dpl)
        logits_gnn = self.class_logits(z_gnn)

        z_cat = torch.cat([z_dpl, z_gnn], dim=-1)
        logits_fuse = self.fuse_head(z_cat)

        return {
            'z_dpl': z_dpl,
            'z_gnn': z_gnn,
            'logits_dpl': logits_dpl,
            'logits_gnn': logits_gnn,
            'logits_fuse': logits_fuse,
        }

    def loss_components(
        self,
        out: Dict[str, torch.Tensor],
        y: torch.Tensor,
        margin: float = 0.1,
        align_weight: float = 0.2,
        proto_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ce_fuse = F.cross_entropy(out['logits_fuse'], y)
        ce_dpl = F.cross_entropy(out['logits_dpl'], y)
        ce_gnn = F.cross_entropy(out['logits_gnn'], y)

        # 细粒度：最近同类原型相似度 vs 最近异类原型相似度
        pos_sim, neg_sim = self.proto_bank(out['z_gnn'].detach(), y)  # 使用当前原型库计算
        # 若某类当前无原型，跳过该样本（mask）
        valid = (pos_sim >= 0) & (neg_sim >= 0)
        if valid.any():
            fg_loss = F.relu(margin - pos_sim[valid] + neg_sim[valid]).mean()
        else:
            fg_loss = torch.tensor(0.0, device=y.device)

        # 分支对齐
        align = (1 - F.cosine_similarity(out['z_dpl'], out['z_gnn'], dim=-1)).mean()

        total = ce_fuse + ce_dpl + ce_gnn + proto_weight * fg_loss + align_weight * align
        return total, {
            'ce_fuse': ce_fuse.detach(),
            'ce_dpl': ce_dpl.detach(),
            'ce_gnn': ce_gnn.detach(),
            'fg_loss': fg_loss.detach(),
            'align': align.detach(),
        }


class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, use_bias=True):
        super(GraphSAGELayer, self).__init__()
        self.linear_neigh = nn.Linear(in_features, out_features, bias=use_bias)
        self.linear_self = nn.Linear(in_features, out_features, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        nn.init.xavier_uniform_(self.linear_neigh.weight)
        nn.init.xavier_uniform_(self.linear_self.weight)
        if use_bias:
            nn.init.constant_(self.linear_neigh.bias, 0)
            nn.init.constant_(self.linear_self.bias, 0)

    def forward(self, x, adj):
        deg = torch.sum(adj, dim=1, keepdim=True)
        deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        neigh_feat = torch.matmul(adj, x) * deg_inv
        neigh_feat = self.linear_neigh(neigh_feat)
        self_feat = self.linear_self(x)
        output = self_feat + neigh_feat
        output = self.activation(output)
        output = self.dropout(output)
        return output


class GraphSAGENet(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.1):
        super(GraphSAGENet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGELayer(in_dim, hidden_dims[0], dropout))
        for i in range(1, len(hidden_dims)):
            self.layers.append(GraphSAGELayer(hidden_dims[i - 1], hidden_dims[i], dropout))

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                if len(param) == hidden_dim * 4:
                    param.data[hidden_dim:hidden_dim * 2] = 1

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        return output, hidden


class GNNConfig:
    num_classes = 11
    signal_length = 128
    patch_length = 64
    patch_stride = 32
    window_size = 8
    gnn_hidden_dims = [64, 64]
    gnn_dropout = 0.1
    rnn_hidden_dim = 128
    rnn_layers = 2
    rnn_dropout = 0.1
    rnn_bidirectional = False


class DTSG(nn.Module):
    def __init__(self, patch_length, patch_stride, window_size):
        super(DTSG, self).__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.window_size = window_size
        self.edge_weights = nn.Parameter(torch.randn(patch_length, patch_length))

    def forward(self, I, Q):
        L = I.shape[0]
        P = math.floor((L - self.patch_length) / self.patch_stride) + 1
        patches = []
        adjs = []
        for n in range(P):
            s = n * self.patch_stride
            e = s + self.patch_length
            I_patch = I[s:e]
            Q_patch = Q[s:e]
            node_features = torch.stack([I_patch, Q_patch], dim=1)
            adj = self._build_adjacency_matrix(self.patch_length, self.window_size)
            patches.append(node_features)
            adjs.append(adj)
        return patches, adjs

    def _build_adjacency_matrix(self, length, window_size):
        adj = torch.zeros(length, length)
        for i in range(length):
            for j in range(max(0, i - window_size), min(length, i + window_size + 1)):
                if i != j:
                    weight = torch.sigmoid(self.edge_weights[i, j])
                    adj[i, j] = weight
        adj = adj * (1 - torch.eye(length))
        return adj


class DTSGNet(nn.Module):
    def __init__(self, config: GNNConfig):
        super(DTSGNet, self).__init__()
        self.config = config
        self.dtsg = DTSG(config.patch_length, config.patch_stride, config.window_size)
        self.gnn = GraphSAGENet(in_dim=2, hidden_dims=config.gnn_hidden_dims, dropout=config.gnn_dropout)
        self.rnn = LSTMEncoder(
            input_dim=config.gnn_hidden_dims[-1],
            hidden_dim=config.rnn_hidden_dim,
            num_layers=config.rnn_layers,
            dropout=config.rnn_dropout,
            bidirectional=config.rnn_bidirectional,
        )
        rnn_output_dim = config.rnn_hidden_dim * (2 if config.rnn_bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, config.num_classes),
        )

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, I, Q):
        batch_size = I.shape[0] if I.dim() > 1 else 1
        if batch_size == 1:
            return self._forward_single(I, Q)
        logits_batch = []
        for i in range(batch_size):
            logits_batch.append(self._forward_single(I[i], Q[i]))
        return torch.cat(logits_batch, dim=0)

    def _forward_single(self, I, Q):
        patches, adjs = self.dtsg(I, Q)
        patch_features = []
        for patch, adj in zip(patches, adjs):
            graph_features = self.gnn(patch, adj)
            patch_feature = torch.sum(graph_features, dim=0)
            patch_features.append(patch_feature)
        if not patch_features:
            patch_features = [torch.zeros(self.config.gnn_hidden_dims[-1]).to(I.device)]
        feature_matrix = torch.stack(patch_features).unsqueeze(0)
        _, rnn_hidden = self.rnn(feature_matrix)
        logits = self.classifier(rnn_hidden)
        return logits

    def forward_features(self, I, Q):
        batch_size = I.shape[0] if I.dim() > 1 else 1
        if batch_size == 1:
            patches, adjs = self.dtsg(I, Q)
            patch_features = []
            for patch, adj in zip(patches, adjs):
                graph_features = self.gnn(patch, adj)
                patch_feature = torch.sum(graph_features, dim=0)
                patch_features.append(patch_feature)
            if not patch_features:
                patch_features = [torch.zeros(self.config.gnn_hidden_dims[-1]).to(I.device)]
            feature_matrix = torch.stack(patch_features).unsqueeze(0)
            _, rnn_hidden = self.rnn(feature_matrix)
            return rnn_hidden
        feats = []
        for i in range(batch_size):
            feats.append(self.forward_features(I[i], Q[i]))
        return torch.cat(feats, dim=0)

class DualBranchTrainer:
    def __init__(
        self,
        model: DualBranchNet,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def train_epoch(self, loader: DataLoader, device: torch.device) -> Dict[str, float]:
        self.model.train()
        loss_sum = 0.0
        n = 0
        meters = {'ce_fuse': 0.0, 'ce_dpl': 0.0, 'ce_gnn': 0.0, 'fg_loss': 0.0, 'align': 0.0}

        for batch in loader:
            # 期望 batch 包含：'img'（DPL 输入），'I', 'Q'（图网络输入），'label'
            img = batch['img'].to(device)
            I = batch['I'].to(device)
            Q = batch['Q'].to(device)
            y = batch['label'].to(device)

            out = self.model(img, I, Q)
            total_loss, parts = self.model.loss_components(out, y)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 在线更新细粒度原型（用 z_gnn ）
            with torch.no_grad():
                self.model.proto_bank.update_online(out['z_gnn'].detach(), y)

            bs = y.shape[0]
            loss_sum += float(total_loss.item()) * bs
            n += bs
            for k in meters:
                meters[k] += float(parts[k].item()) * bs

        self.scheduler.step()
        for k in meters:
            meters[k] /= max(1, n)
        meters['loss'] = loss_sum / max(1, n)
        return meters

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader, device: torch.device) -> Dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        for batch in loader:
            img = batch['img'].to(device)
            I = batch['I'].to(device)
            Q = batch['Q'].to(device)
            y = batch['label'].to(device)

            out = self.model(img, I, Q)
            logits = out['logits_fuse']
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

        acc = correct / max(1, total)
        return {'acc': acc}


def build_dual_branch(
    num_classes: int,
    dpl_feat_dim: int = 128,
    temperature: float = 0.5,
    max_prototypes_per_class: int = 5,
    dp_lambda: float = 0.7,
    ema_momentum: float = 0.9,
    merge_threshold: float = 0.2,
) -> DualBranchNet:
    gcfg = GNNConfig()
    bank = PrototypeBank(
        num_classes=num_classes,
        feat_dim=dpl_feat_dim,
        max_prototypes_per_class=max_prototypes_per_class,
        dp_lambda=dp_lambda,
        ema_momentum=ema_momentum,
        merge_threshold=merge_threshold,
    )
    model = DualBranchNet(
        num_classes=num_classes,
        dpl_feat_dim=dpl_feat_dim,
        gnn_config=gcfg,
        temperature=temperature,
        proto_bank=bank,
    )
    return model


def example_train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: Optional[torch.device] = None,
    epochs: int = 50,
):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_dual_branch(num_classes=num_classes)
    model = model.to(device)

    trainer = DualBranchTrainer(model, epochs=epochs)

    for e in range(1, epochs + 1):
        meters = trainer.train_epoch(train_loader, device)
        evalm = trainer.eval_epoch(val_loader, device)
        print(f"Epoch {e:03d} | loss {meters['loss']:.4f} | ce_fuse {meters['ce_fuse']:.4f} | ce_dpl {meters['ce_dpl']:.4f} | ce_gnn {meters['ce_gnn']:.4f} | fg {meters['fg_loss']:.4f} | align {meters['align']:.4f} | val_acc {evalm['acc']:.4f}")


if __name__ == '__main__':
    print('This file defines the dual-branch model and a training example.\n'
          'To run example_train, provide DataLoaders yielding dicts with keys: img, I, Q, label.')


