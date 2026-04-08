import os
import sys
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, global_mean_pool
from torch_geometric.data import Data, Batch

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from openood.utils import config
from openood.datasets import get_dataloader

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

# Random seed
os.environ['PYTHONHASHSEED'] = str(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Config_gnn:
    """DTSG-Net configuration"""
    def __init__(self):
        self.num_classes = 8
        self.epochs = 100
        self.batch_size = 256
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.save_dir = "results/dtsg_rml22"
        self.device = device

        self.config_files = [
            './configs/datasets/rml22/rml22.yml',
            './configs/datasets/rml22/rml22_ood.yml',
            './configs/networks/resnet18_32x32.yml',
            './configs/pipelines/test/test_ood.yml',
            './configs/preprocessors/base_preprocessor.yml',
            './configs/postprocessors/msp.yml',
        ]

        # DTSG-Net
        self.signal_length = 128
        self.patch_length = 8
        self.patch_stride = 4
        self.local_window = 4
        self.gnn_hidden_dim = 128
        self.gnn_layers = 3
        self.rnn_hidden_dim = 128
        self.rnn_layers = 2
        self.rnn_type = 'GRU'
        self.dropout = 0.1


class ETFSemanticHead(nn.Module):
    """ETF as Shared Semantic Coordinate System"""
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.etf_matrix = self._build_etf_matrix(feat_dim, num_classes).to(device)

    def _build_etf_matrix(self, feat_dim, num_classes):
        P = torch.randn(feat_dim, num_classes)
        Q, _ = torch.linalg.qr(P)
        I = torch.eye(num_classes)
        J = torch.ones(num_classes, num_classes)
        return np.sqrt(num_classes / (num_classes - 1)) * Q @ (I - J / num_classes)

    def forward(self, h_global):
        return h_global @ self.etf_matrix

    def contrastive_loss(self, patch_feat, target, num_patches, temperature=0.1):
        """Align patch features to class ETF prototypes"""
        num_patches_per_sample = patch_feat.shape[0] // target.shape[0]
        prototypes = self.etf_matrix.T
        patch_norm = F.normalize(patch_feat, dim=-1)
        proto_norm = F.normalize(prototypes, dim=0).t()
        sim = patch_norm @ proto_norm
        target_exp = target.unsqueeze(1).expand(-1, num_patches_per_sample).reshape(-1)
        return F.cross_entropy(sim / temperature, target_exp)


class DTSG(nn.Module):
    """Dynamic Topological Signal Graph: Signal -> Patch Graphs"""
    def __init__(self, patch_length=32, patch_stride=32, local_window=8, signal_length=128):
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.local_window = local_window
        self.num_patches = ((signal_length - patch_length) // patch_stride) + 1
        self.edge_weights = nn.Parameter(torch.randn(local_window))
        self.register_buffer('edge_index', self._precompute_edge_index())
        self.register_buffer('edge_distance', self._precompute_edge_distance())

    def _precompute_edge_index(self):
        edge_list = [[i, j] for i in range(self.patch_length) for j in range(self.patch_length)
                      if 0 < abs(i - j) <= self.local_window]
        return torch.tensor(edge_list or [[i, i] for i in range(self.patch_length)],
                           dtype=torch.long).t().contiguous()

    def _precompute_edge_distance(self):
        distances = [abs(i - j) - 1 for i in range(self.patch_length) for j in range(self.patch_length)
                     if 0 < abs(i - j) <= self.local_window]
        return torch.tensor(distances or [0] * self.patch_length, dtype=torch.long)

    def forward(self, iq_signal):
        if iq_signal.dim() == 3 and iq_signal.shape[1] == 2:
            iq_signal = iq_signal.transpose(1, 2)  # [B, 2, L] -> [B, L, 2]
        batch_size = iq_signal.shape[0]
        device = iq_signal.device

        # Vectorized patch extraction using unfold
        iq_signal_t = iq_signal.transpose(1, 2)  # [B, L, 2] -> [B, 2, L]
        patches_list = []
        for c in range(2):
            signal_2d = iq_signal_t[:, c:c+1, :].unsqueeze(2)  # [B, 1, 1, L]
            unfolded = F.unfold(signal_2d, kernel_size=(1, self.patch_length), stride=(1, self.patch_stride))
            patches_list.append(unfolded.transpose(1, 2))  # [B, num_patches, patch_length]
        patches = torch.stack(patches_list, dim=-1)  # [B, num_patches, patch_length, 2]

        # Flatten: [B * num_patches * patch_length, 2]
        patches_flat = patches.reshape(batch_size * self.num_patches, self.patch_length, 2)
        node_features = patches_flat.reshape(-1, 2)

        # Vectorized batch edge_index
        num_edges = self.edge_index.shape[1]
        num_graphs = batch_size * self.num_patches
        num_nodes_per_graph = self.patch_length
        graph_offsets = torch.arange(0, num_graphs * num_nodes_per_graph, num_nodes_per_graph, device=device)
        offsets = graph_offsets.view(-1, 1)
        edge_index_base = self.edge_index.to(device)
        edge_index_expanded = edge_index_base.unsqueeze(0).expand(num_graphs, -1, -1)
        batch_edge_index_per_graph = edge_index_expanded + offsets.unsqueeze(1)
        batch_edge_index = batch_edge_index_per_graph.permute(1, 0, 2).reshape(2, -1)

        # Edge attributes
        edge_attr = self.edge_weights[self.edge_distance]
        batch_edge_attr = edge_attr.repeat(num_graphs)

        # Batch vector
        batch = torch.arange(num_graphs, device=device).repeat_interleave(num_nodes_per_graph)

        # Direct Batch construction (no from_data_list)
        return Batch(x=node_features, edge_index=batch_edge_index, edge_attr=batch_edge_attr, batch=batch)


class PatchGNN(nn.Module):
    """GraphSAGE for patch-level feature encoding"""
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            GraphSAGE(input_dim if i == 0 else hidden_dim, hidden_dim, num_layers=1)
            for i in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_graph):
        x = batch_graph.x
        edge_index = batch_graph.edge_index
        edge_weight = getattr(batch_graph, 'edge_attr', None)
        for i, (gnn, ln) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            x_new = self.dropout(F.relu(ln(gnn(x, edge_index, edge_weight=edge_weight))))
            if i > 0 and x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        return global_mean_pool(x, batch_graph.batch)


class TemporalRNN(nn.Module):
    """RNN for temporal modeling with attention-based global pooling"""
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2, rnn_type='GRU', dropout=0.1):
        super().__init__()
        RNN = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = RNN(input_dim, hidden_dim, num_layers, batch_first=True,
                       dropout=dropout if num_layers > 1 else 0, bidirectional=False)

    def forward(self, patch_features, num_patches):
        batch_size = patch_features.shape[0] // num_patches
        seq = patch_features.view(batch_size, num_patches, -1)
        output, _ = self.rnn(seq)
        # attn_weights = F.softmax(self.attn(output), dim=1).squeeze(-1)
        # g = (output * attn_weights.unsqueeze(-1)).sum(dim=1)
        g = output[:, -1, :]  # 直接取最后一个时刻
        return g


class DTSGNet(nn.Module):
    """DTSG-Net with Hierarchical ETF Semantic Consistency"""
    def __init__(self, num_classes=8, signal_length=128, patch_length=8, patch_stride=4,
                 local_window=4, gnn_hidden_dim=128, gnn_layers=3, rnn_hidden_dim=128,
                 rnn_layers=2, rnn_type='GRU', dropout=0.1):
        super().__init__()
        self.dtsg = DTSG(patch_length, patch_stride, local_window, signal_length)
        self.num_patches = self.dtsg.num_patches
        self.patch_gnn = PatchGNN(2, gnn_hidden_dim, gnn_layers, dropout)
        self.temporal_rnn = TemporalRNN(gnn_hidden_dim, rnn_hidden_dim, rnn_layers, rnn_type, dropout)
        self.etf_semantic = ETFSemanticHead(gnn_hidden_dim, num_classes)

    def forward(self, iq_signal):
        batch_graph = self.dtsg(iq_signal)  # returns Batch directly
        patch_features = self.patch_gnn(batch_graph)
        global_features = self.temporal_rnn(patch_features, self.num_patches)
        logits_g = self.etf_semantic(global_features)
        return {'logits_g': logits_g, 'global_features': global_features, 'patch_features': patch_features}

    def compute_losses(self, iq_signal, target):
        outputs = self.forward(iq_signal)
        logits_g = outputs['logits_g']
        patch_feat = outputs['patch_features']

        ce_g = F.cross_entropy(logits_g, target)
        loss_contrast = self.etf_semantic.contrastive_loss(patch_feat, target, self.num_patches)

        loss = ce_g + 0.1 * loss_contrast
        return loss, {'ce_g': ce_g.item(), 'contrast': loss_contrast.item()}, logits_g

class DTSGTrainer:
    def __init__(self, net, config_gnn, train_loader, val_loader):
        self.net = net
        self.config_gnn = config_gnn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config_gnn.lr, weight_decay=config_gnn.weight_decay)
        total_steps = config_gnn.epochs * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            lr_lambda=lambda s: 1e-6/config_gnn.lr + (1 - 1e-6/config_gnn.lr) * (1 + np.cos(np.pi*s/total_steps))/2)

    def train_epoch(self, epoch_idx):
        self.net.train()
        total_loss, ce_g_loss, contrast_loss, correct, total = 0.0, 0.0, 0.0, 0, 0
        for batch in tqdm(self.train_loader, desc=f'DTSG {epoch_idx:03d}'):
            data, target = batch['data'].to(self.config_gnn.device).squeeze(1), batch['label'].to(self.config_gnn.device)
            self.optimizer.zero_grad()
            loss, loss_dict, logits = self.net.compute_losses(data, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            ce_g_loss = ce_g_loss * 0.9 + loss_dict['ce_g'] * 0.1
            contrast_loss = contrast_loss * 0.9 + loss_dict['contrast'] * 0.1
            total_loss = total_loss * 0.9 + loss.item() * 0.1
            correct += logits.argmax(1).eq(target).sum().item()
            total += target.size(0)
        return {'epoch_idx': epoch_idx, 'total_loss': total_loss, 'ce_g': ce_g_loss, 'contrast': contrast_loss,
                'acc': correct/total, 'lr': self.optimizer.param_groups[0]['lr']}

    def validate(self, epoch_idx):
        self.net.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Val {epoch_idx:03d}'):
                data, target = batch['data'].to(self.config_gnn.device).squeeze(1), batch['label'].to(self.config_gnn.device)
                outputs = self.net(data)
                logits = outputs['logits_g']
                total_loss += F.cross_entropy(logits, target).item()
                correct += logits.argmax(1).eq(target).sum().item()
                total += target.size(0)
        return {'loss': total_loss/len(self.val_loader), 'acc': correct/total if total > 0 else 0.0}

class DTSGTrainingManager:
    def __init__(self, config_gnn, model):
        self.config_gnn = config_gnn
        self.device = config_gnn.device
        self.model = model.to(self.device)
        self.best_acc = 0.0
        self.best_epoch_idx = 0

    def train(self, train_loader, val_loader):
        trainer = DTSGTrainer(self.model, self.config_gnn, train_loader, val_loader)
        for epoch in range(1, self.config_gnn.epochs + 1):
            train_metrics = trainer.train_epoch(epoch)
            val_metrics = trainer.validate(epoch)
            self.save_model(epoch, val_metrics, self.config_gnn.epochs)
            print(f'  Ep {epoch:03d} | Train {train_metrics["total_loss"]:.4f} | Val {val_metrics["loss"]:.4f} | '
                  f'Acc {val_metrics["acc"]:.4f} | CE_g {train_metrics["ce_g"]:.4f} | Contrast {train_metrics["contrast"]:.4f}')
        return self.model, trainer

    def save_model(self, epoch_idx, val_metrics, num_epochs):
        output_dir = self.config_gnn.save_dir
        os.makedirs(output_dir, exist_ok=True)
        try:
            state_dict = self.model.module.state_dict()
        except AttributeError:
            state_dict = self.model.state_dict()
        if val_metrics['acc'] >= self.best_acc:
            if self.best_epoch_idx > 0:
                Path(os.path.join(output_dir, f'best_epoch{self.best_epoch_idx}_acc{self.best_acc:.4f}.ckpt')).unlink(missing_ok=True)
            self.best_epoch_idx, self.best_acc = epoch_idx, val_metrics['acc']
            torch.save(state_dict, os.path.join(output_dir, f'best_epoch{self.best_epoch_idx}_acc{self.best_acc:.4f}.ckpt'))
            torch.save(state_dict, os.path.join(output_dir, 'best.ckpt'))
        if epoch_idx == num_epochs:
            torch.save(state_dict, os.path.join(output_dir, f'last_epoch{epoch_idx}_acc{val_metrics["acc"]:.4f}.ckpt'))

def load_checkpoint(model, path, map_location=None):
    sd = torch.load(path, map_location=map_location)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    return model

def get_predictions(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predict'):
            data = batch['data'].to(device).squeeze(1)
            outputs = model(data)
            _, pred = torch.max(outputs['logits_g'], 1)
            preds.extend(pred.cpu().numpy())
            labels.extend(batch['label'].numpy())
    return np.array(preds), np.array(labels)

def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix', save_path=None, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 3)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(np.arange(len(class_names or [])), class_names, rotation=45, ha='right')
    plt.yticks(np.arange(len(class_names or [])), class_names)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return cm, np.trace(cm) / np.sum(cm)

def test_and_plot_confusion_matrix(checkpoint_path=None, class_names=None, normalize=True, model=None):
    config_gnn = Config_gnn()
    loader_dict = get_dataloader(config.Config(*config_gnn.config_files))
    test_loader = loader_dict['test']
    if model is None:
        model = DTSGNet(config_gnn.num_classes, config_gnn.signal_length, config_gnn.patch_length,
                        config_gnn.patch_stride, config_gnn.local_window, config_gnn.gnn_hidden_dim,
                        config_gnn.gnn_layers, config_gnn.rnn_hidden_dim, config_gnn.rnn_layers,
                        config_gnn.rnn_type, config_gnn.dropout).to(config_gnn.device)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config_gnn.save_dir, 'best.ckpt')
    if os.path.exists(checkpoint_path):
        model = load_checkpoint(model, checkpoint_path, config_gnn.device)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    if class_names is None:
        class_names = ['QAM16', 'QAM64', '8PSK', 'QPSK', 'BPSK', 'AM-DSB', 'AM-SSB', 'WBFM']
    save_path = os.path.join(config_gnn.save_dir, 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png')
    y_pred, y_true = get_predictions(model, test_loader, config_gnn.device)
    return plot_confusion_matrix(y_true, y_pred, class_names, save_path=save_path, normalize=normalize)


if __name__ == '__main__':
    config_gnn = Config_gnn()
    config_openood = config.Config(*config_gnn.config_files)
    os.makedirs(config_gnn.save_dir, exist_ok=True)
    loader_dict = get_dataloader(config_openood)

    model = DTSGNet(config_gnn.num_classes, config_gnn.signal_length, config_gnn.patch_length,
                    config_gnn.patch_stride, config_gnn.local_window, config_gnn.gnn_hidden_dim,
                    config_gnn.gnn_layers, config_gnn.rnn_hidden_dim, config_gnn.rnn_layers,
                    config_gnn.rnn_type, config_gnn.dropout)

    manager = DTSGTrainingManager(config_gnn, model)
    model, trainer = manager.train(loader_dict['train'], loader_dict['val'])

    test_and_plot_confusion_matrix(normalize=True)
    print(f"\nDone! Best: {manager.best_acc:.4f} @ epoch {manager.best_epoch_idx}")
