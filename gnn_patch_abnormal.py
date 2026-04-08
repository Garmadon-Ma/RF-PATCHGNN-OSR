import os
import sys
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, global_mean_pool
from torch_geometric.data import Batch

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.metrics import compute_all_metrics

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
    def __init__(self):
        self.num_classes = 8
        self.epochs = 100
        self.batch_size = 256
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.save_dir = "results/rtsg_unified_graph"
        self.device = device

        self.config_files = [
            './configs/datasets/rml22/rml22.yml',
            './configs/datasets/rml22/rml22_ood.yml',
            './configs/networks/resnet18_32x32.yml',
            './configs/pipelines/test/test_ood.yml',
            './configs/preprocessors/base_preprocessor.yml',
            './configs/postprocessors/msp.yml',
        ]

        # RTSG-Net config
        self.signal_length = 128
        self.patch_length = 16
        self.patch_stride = 4
        self.local_window = 4
        self.gnn_hidden_dim = 256
        self.gnn_layers = 4
        self.dropout = 0.1

class DTSG(nn.Module):
    """Signal -> Unified Graph with intra-patch and cross-patch edges."""
    def __init__(self, patch_length=16, patch_stride=4, local_window=4, signal_length=128):
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.local_window = local_window
        self.signal_length = signal_length
        self.num_patches = ((signal_length - patch_length) // patch_stride) + 1
        
        # Patch内边权重（距离越近权重越大）
        self.edge_weights = nn.Parameter(torch.randn(local_window))
        # 跨Patch边权重（可学习）
        self.cross_patch_weight = nn.Parameter(torch.tensor(1.0))
        
    def _precompute_intra_edge_index(self):
        edge_list = [[i, j] for i in range(self.patch_length) for j in range(self.patch_length)
                      if 0 < abs(i - j) <= self.local_window]
        return torch.tensor(edge_list or [[i, i] for i in range(self.patch_length)],
                           dtype=torch.long).t().contiguous()

    def _precompute_intra_edge_distance(self):
        distances = [abs(i - j) - 1 for i in range(self.patch_length) for j in range(self.patch_length)
                     if 0 < abs(i - j) <= self.local_window]
        return torch.tensor(distances or [0] * self.patch_length, dtype=torch.long)

    def forward(self, iq_signal):
        if iq_signal.dim() == 5:
            iq_signal = iq_signal.reshape(iq_signal.shape[0], 2, -1, iq_signal.shape[-1])
            iq_signal = iq_signal.squeeze(2)
            iq_signal = iq_signal.transpose(1, 2)
        elif iq_signal.dim() == 3 and iq_signal.shape[1] == 2:
            iq_signal = iq_signal.transpose(1, 2)
        
        batch_size = iq_signal.shape[0]
        signal_len = iq_signal.shape[2]
        device = iq_signal.device
        
        # Dynamically compute num_patches based on actual signal length
        self.num_patches = ((signal_len - self.patch_length) // self.patch_stride) + 1
        
        iq_signal_t = iq_signal.transpose(1, 2)
        patches_list = []
        for c in range(2):
            signal_2d = iq_signal_t[:, c:c+1, :].unsqueeze(2)
            unfolded = F.unfold(signal_2d, kernel_size=(1, self.patch_length), stride=(1, self.patch_stride))
            patches_list.append(unfolded.transpose(1, 2))
        patches = torch.stack(patches_list, dim=-1)
        
        patches_flat = patches.reshape(batch_size * self.num_patches, self.patch_length, 2)
        node_features = patches_flat.reshape(-1, 2)
        
        num_graphs = batch_size * self.num_patches
        
        # Intra-patch edges
        intra_edge_index_base = self._precompute_intra_edge_index().to(device)
        intra_edge_dist_base = self._precompute_intra_edge_distance().to(device)
        
        graph_offsets = torch.arange(0, num_graphs * self.patch_length, self.patch_length, device=device)
        offsets = graph_offsets.view(-1, 1)
        
        intra_edge_expanded = intra_edge_index_base.unsqueeze(0).expand(num_graphs, -1, -1)
        intra_batch_edge = intra_edge_expanded + offsets.unsqueeze(1)
        intra_edge_index = intra_batch_edge.permute(1, 0, 2).reshape(2, -1)
        
        intra_edge_attr = self.edge_weights[intra_edge_dist_base]
        intra_edge_attr_batch = intra_edge_attr.repeat(num_graphs)
        
        # Cross-patch edges
        cross_src, cross_dst = [], []
        for b in range(batch_size):
            for p in range(self.num_patches - 1):
                src_node = b * self.num_patches * self.patch_length + p * self.patch_length + (self.patch_length - 1)
                dst_node = b * self.num_patches * self.patch_length + (p + 1) * self.patch_length
                cross_src.append(src_node)
                cross_dst.append(dst_node)
        
        cross_edge_index = torch.tensor([cross_src, cross_dst], dtype=torch.long, device=device)
        cross_edge_attr = self.cross_patch_weight.expand(len(cross_src))
        
        # Combine all edges
        all_edge_index = torch.cat([intra_edge_index, cross_edge_index], dim=1)
        all_edge_attr = torch.cat([intra_edge_attr_batch, cross_edge_attr], dim=0)
        
        batch = torch.arange(num_graphs, device=device).repeat_interleave(self.patch_length)
        
        return Batch(x=node_features, edge_index=all_edge_index, edge_attr=all_edge_attr, batch=batch)

class PatchGNN(nn.Module):
    """GraphSAGE for node-level encoding (no global pooling)."""
    def __init__(self, input_dim=2, hidden_dim=256, num_layers=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
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

        return x  # node-level embeddings, shape: [total_nodes, dim]

class PatchEdgeLabeller(nn.Module):
    """Edge anomaly detection - matches EdgeLabellerFusedOri from NSReg."""
    def __init__(self, feat_dim=256, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim
        self.weight = nn.Parameter(torch.Tensor(feat_dim, feat_dim))
        nn.init.kaiming_uniform_(self.weight)
        self.linear = nn.Linear(feat_dim, 1)

    def forward(self, feat1, feat2):
        fused = torch.matmul(torch.sigmoid(feat1), self.weight) * torch.sigmoid(feat2)
        return self.linear(fused)


class RTSGNet(nn.Module):
    """Unified RTSGNet with cross-patch graph edges.
    Follows NSReg style: use node embeddings directly for edge detection.
    """
    def __init__(self, num_classes=8, signal_length=128, patch_length=16, patch_stride=4,
                 local_window=4, gnn_hidden_dim=256, gnn_layers=4, dropout=0.1):
        super().__init__()

        self.dtsg = DTSG(patch_length, patch_stride, local_window, signal_length)
        self.num_patches = self.dtsg.num_patches
        self.patch_length = patch_length
        self.signal_length = signal_length
        self.patch_gnn = PatchGNN(2, gnn_hidden_dim, gnn_layers, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.LayerNorm(gnn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, num_classes)
        )

        self.edge_labeller = PatchEdgeLabeller(feat_dim=gnn_hidden_dim)

    def _get_global_features(self, node_embeddings, batch_idx):
        """Pool node embeddings per graph -> global features for classification."""
        # node_embeddings: [total_nodes, dim], batch_idx: [total_nodes]
        return global_mean_pool(node_embeddings, batch_idx)

    def _get_patch_representatives(self, node_embeddings, batch_size):
        """Get the LAST node of each patch as patch representative.
        Cross-patch edges in DTSG connect patch_last -> next_patch_first.
        """
        total_patches = batch_size * self.num_patches
        idx = []
        for b in range(batch_size):
            for p in range(self.num_patches):
                # Last node of patch p in batch b
                node_id = b * self.num_patches * self.patch_length + p * self.patch_length + (self.patch_length - 1)
                idx.append(node_id)
        return node_embeddings[torch.tensor(idx, device=node_embeddings.device, dtype=torch.long)]

    def forward(self, iq_signal, return_edge_scores=False):
        if iq_signal.dim() == 5:
            iq_signal = iq_signal.reshape(iq_signal.shape[0], 2, -1, iq_signal.shape[-1])
            iq_signal = iq_signal.squeeze(2)
            iq_signal = iq_signal.transpose(1, 2)
        elif iq_signal.dim() == 3 and iq_signal.shape[1] == 2:
            iq_signal = iq_signal.transpose(1, 2)

        batch_graph = self.dtsg(iq_signal)
        self.num_patches = self.dtsg.num_patches
        node_embeddings = self.patch_gnn(batch_graph)  # [total_nodes, dim]

        batch_size = iq_signal.shape[0]
        global_features = self._get_global_features(node_embeddings, batch_graph.batch)
        logits = self.classifier(global_features)

        outputs = {
            'logits': logits,
            'global_features': global_features,
            'node_embeddings': node_embeddings,
            'batch_idx': batch_graph.batch
        }

        if return_edge_scores:
            patch_repr = self._get_patch_representatives(node_embeddings, batch_size)
            outputs['patch_repr'] = patch_repr
            edge_scores = self._compute_edge_anomaly_scores(patch_repr, batch_size)
            outputs['edge_anomaly_scores'] = edge_scores

        return outputs

    def _compute_edge_anomaly_scores(self, patch_repr, batch_size):
        """Compute edge anomaly scores using EdgeLabeller on patch representatives."""
        device = patch_repr.device
        all_edge_scores = []
        for b in range(batch_size):
            base = b * self.num_patches
            src_idx = torch.tensor([base + p for p in range(self.num_patches - 1)],
                                   device=device, dtype=torch.long)
            dst_idx = torch.tensor([base + p for p in range(1, self.num_patches)],
                                   device=device, dtype=torch.long)

            scores = self.edge_labeller(patch_repr[src_idx], patch_repr[dst_idx])
            all_edge_scores.append(scores.mean())

        return torch.stack(all_edge_scores)

    def _compute_edge_loss(self, node_embeddings, batch_size):
        """Edge loss: positive edges (adjacent patches) vs negative edges (rewired).
        Uses EdgeLabeller like NSReg."""
        patch_repr = self._get_patch_representatives(node_embeddings, batch_size)
        device = patch_repr.device
        pos_edges_src, pos_edges_dst = [], []
        for b in range(batch_size):
            for p in range(self.num_patches - 1):
                base = b * self.num_patches
                pos_edges_src.append(base + p)
                pos_edges_dst.append(base + p + 1)

        if len(pos_edges_src) == 0:
            return torch.tensor(0.0, device=device)

        pos_edges_src = torch.tensor(pos_edges_src, device=device, dtype=torch.long)
        pos_edges_dst = torch.tensor(pos_edges_dst, device=device, dtype=torch.long)

        perm_idx = torch.randperm(len(pos_edges_src), device=device)
        neg_edges_src = pos_edges_src.clone()
        neg_edges_dst = pos_edges_src[perm_idx]

        pos_feat_src = patch_repr[pos_edges_src]
        pos_feat_dst = patch_repr[pos_edges_dst]
        neg_feat_src = patch_repr[neg_edges_src]
        neg_feat_dst = patch_repr[neg_edges_dst]

        pos_scores = self.edge_labeller(pos_feat_src, pos_feat_dst)
        neg_scores = self.edge_labeller(neg_feat_src, neg_feat_dst)

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_scores, neg_scores]),
            torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).squeeze(-1)
        )
        return loss

    def compute_losses(self, iq_signal, target):
        outputs = self.forward(iq_signal, return_edge_scores=True)
        logits = outputs['logits']
        node_emb = outputs['node_embeddings']

        ce_loss = F.cross_entropy(logits, target)
        edge_loss = self._compute_edge_loss(node_emb, target.shape[0])

        total_loss = ce_loss + 0.5 * edge_loss

        return total_loss, {
            'ce': ce_loss.item(),
            'edge': edge_loss.item()
        }, logits


class RTSGEdgeOODPostprocessor:
    """Edge-based OOD detection using patch-to-patch anomaly scores."""
    
    def __init__(self):
        self.setup_flag = False

    def setup(self, net, id_train_loader, device):
        if self.setup_flag:
            return
            
        net.eval()
        edge_scores_list = []
        
        with torch.no_grad():
            for batch in tqdm(id_train_loader, desc='Edge OOD Setup'):
                data = batch['data'].to(device).squeeze(1)
                outputs = net(data, return_edge_scores=True)
                edge_scores = outputs.get('edge_anomaly_scores', torch.zeros(data.shape[0], device=device))
                edge_scores_list.append(edge_scores.cpu())
        
        all_edge_scores = torch.cat(edge_scores_list)
        self.edge_scores_mean = all_edge_scores.mean().item()
        self.edge_scores_std = all_edge_scores.std().item()
        self.setup_flag = True
        print(f'Edge OOD Setup: mean={self.edge_scores_mean:.4f}, std={self.edge_scores_std:.4f}')

    @torch.no_grad()
    def postprocess(self, net, data, device):
        net.eval()
        data = data.to(device)
        
        outputs = net(data, return_edge_scores=True)
        logits = outputs['logits']
        edge_scores = outputs.get('edge_anomaly_scores', torch.zeros(data.shape[0], device=device))
        
        ood_score = edge_scores.cpu()
        
        _, pred = torch.max(logits, dim=1)
        
        return pred.cpu(), ood_score

    def inference(self, net, data_loader, device, progress=True):
        """Run inference on a data loader."""
        pred_list, score_list, label_list = [], [], []
        
        net.eval()
        iterator = tqdm(data_loader, desc='Edge OOD Inference') if progress else data_loader
        
        with torch.no_grad():
            for batch in iterator:
                data = batch['data'].to(device).squeeze(1)
                label = batch['label']
                
                pred, ood_score = self.postprocess(net, data, device)
                
                pred_list.append(pred.cpu())
                score_list.append(ood_score.cpu())
                label_list.append(label)
        
        pred_list = torch.cat(pred_list).numpy()
        score_list = torch.cat(score_list).numpy()
        label_list = torch.cat(label_list).numpy()
        
        return pred_list, score_list, label_list

class RTSGTrainer:
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
        total_loss, ce_loss, edge_loss, correct, total = 0.0, 0.0, 0.0, 0, 0
        for batch in tqdm(self.train_loader, desc=f'RTSG {epoch_idx:03d}'):
            data, target = batch['data'].to(self.config_gnn.device).squeeze(1), batch['label'].to(self.config_gnn.device)
            self.optimizer.zero_grad()
            loss, loss_dict, logits = self.net.compute_losses(data, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            ce_loss = ce_loss * 0.9 + loss_dict['ce'] * 0.1
            edge_loss = edge_loss * 0.9 + loss_dict['edge'] * 0.1
            total_loss = total_loss * 0.9 + loss.item() * 0.1
            correct += logits.argmax(1).eq(target).sum().item()
            total += target.size(0)
        return {'epoch_idx': epoch_idx, 'total_loss': total_loss, 'ce': ce_loss,
                'edge': edge_loss, 'acc': correct/total, 'lr': self.optimizer.param_groups[0]['lr']}

    def validate(self, epoch_idx):
        self.net.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Val {epoch_idx:03d}'):
                data, target = batch['data'].to(self.config_gnn.device).squeeze(1), batch['label'].to(self.config_gnn.device)
                outputs = self.net(data)
                logits = outputs['logits']
                total_loss += F.cross_entropy(logits, target).item()
                correct += logits.argmax(1).eq(target).sum().item()
                total += target.size(0)
        return {'loss': total_loss/len(self.val_loader), 'acc': correct/total if total > 0 else 0.0}

class RTSGTrainingManager:
    def __init__(self, config_gnn, model):
        self.config_gnn = config_gnn
        self.device = config_gnn.device
        self.model = model.to(self.device)
        self.best_acc = 0.0
        self.best_epoch_idx = 0

    def train(self, train_loader, val_loader):
        trainer = RTSGTrainer(self.model, self.config_gnn, train_loader, val_loader)
        for epoch in range(1, self.config_gnn.epochs + 1):
            train_metrics = trainer.train_epoch(epoch)
            val_metrics = self.validate(val_loader, epoch)
            self.save_model(epoch, val_metrics, self.config_gnn.epochs)
            print(f'  Ep {epoch:03d} | Train {train_metrics["total_loss"]:.4f} | Val {val_metrics["loss"]:.4f} | '
                  f'Acc {val_metrics["acc"]:.4f} | CE {train_metrics["ce"]:.4f} | EDGE {train_metrics["edge"]:.4f}')
        return self.model, trainer

    def validate(self, val_loader, epoch_idx):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Val {epoch_idx:03d}'):
                data = batch['data'].to(self.device).squeeze(1)
                target = batch['label'].to(self.device)
                outputs = self.model(data)
                logits = outputs['logits']
                total_loss += F.cross_entropy(logits, target).item()
                correct += logits.argmax(1).eq(target).sum().item()
                total += target.size(0)
        return {'loss': total_loss/len(val_loader) if len(val_loader) > 0 else 0.0,
                'acc': correct/total if total > 0 else 0.0}

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


def run_edge_ood_detection(model, test_loader, test_ood_loader, config_gnn, train_loader=None):
    """Run edge-based OOD detection."""
    model.eval()
    
    postprocessor = RTSGEdgeOODPostprocessor()
    if train_loader is not None:
        postprocessor.setup(model, train_loader, config_gnn.device)
    
    id_pred_list, id_conf_list, id_label_list = [], [], []
    ood_pred_list, ood_conf_list, ood_label_list = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Edge OOD ID'):
            data = batch['data'].to(config_gnn.device).squeeze(1)
            label = batch['label']
            pred, ood_score = postprocessor.postprocess(model, data, config_gnn.device)
            id_pred_list.append(pred.cpu())
            id_conf_list.append(ood_score.cpu())
            id_label_list.append(label)
        
        for batch in tqdm(test_ood_loader, desc='Edge OOD OOD'):
            data = batch['data'].to(config_gnn.device).squeeze(1)
            label = batch['label']
            pred, ood_score = postprocessor.postprocess(model, data, config_gnn.device)
            ood_pred_list.append(pred.cpu())
            ood_conf_list.append(ood_score.cpu())
            ood_label_list.append(label)
    
    id_pred_list = torch.cat(id_pred_list).numpy()
    id_conf_list = torch.cat(id_conf_list).numpy()
    id_label_list = torch.cat(id_label_list).numpy()
    ood_pred_list = torch.cat(ood_pred_list).numpy()
    ood_conf_list = torch.cat(ood_conf_list).numpy()
    ood_label_list = torch.cat(ood_label_list).numpy()
    ood_label_list = -1 * np.ones_like(ood_label_list)
    
    pred = np.concatenate([id_pred_list, ood_pred_list])
    conf = np.concatenate([id_conf_list, ood_conf_list])
    label = np.concatenate([id_label_list, ood_label_list])
    
    metrics = compute_all_metrics(conf, label, pred)
    print(
        f"Edge OOD - FPR: {metrics[0]:.4f}, AUROC: {metrics[1]:.4f}, "
        f"AUPR_IN: {metrics[2]:.4f}, AUPR_OUT: {metrics[3]:.4f}, "
        f"ACC: {metrics[4]:.4f}"
    )
    
    return metrics


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
            _, pred = torch.max(outputs['logits'], 1)
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

def test_and_plot_confusion_matrix(model=None, checkpoint_path=None, class_names=None, normalize=True):
    config_gnn = Config_gnn()
    loader_dict = get_dataloader(config.Config(*config_gnn.config_files))    
    save_path = os.path.join(config_gnn.save_dir, 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png')
    
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader_dict['val'], desc='Test'):
            data = batch['data'].to(config_gnn.device).squeeze(1)
            target = batch['label'].to(config_gnn.device)
            outputs = model(data)
            logits = outputs['logits']
            pred = logits.argmax(1)
            preds.extend(pred.cpu().numpy())
            labels.extend(target.cpu().numpy())
    
    y_pred, y_true = np.array(preds), np.array(labels)
    return plot_confusion_matrix(y_true, y_pred, class_names, save_path=save_path, normalize=normalize)


if __name__ == '__main__':
    config_gnn = Config_gnn()
    config_openood = config.Config(*config_gnn.config_files)
    os.makedirs(config_gnn.save_dir, exist_ok=True)
    loader_dict = get_dataloader(config_openood)
    train_loader = loader_dict['train']
    val_loader = loader_dict['val']
    test_loader = loader_dict['test']
    
    ood_loader_dict = get_ood_dataloader(config_openood)
    test_ood_loader = ood_loader_dict['val']

    model = RTSGNet(
        num_classes=config_gnn.num_classes,
        signal_length=config_gnn.signal_length,
        patch_length=config_gnn.patch_length,
        patch_stride=config_gnn.patch_stride,
        local_window=config_gnn.local_window,
        gnn_hidden_dim=config_gnn.gnn_hidden_dim,
        gnn_layers=config_gnn.gnn_layers,
        dropout=config_gnn.dropout
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of patches: {model.num_patches}")

    manager = RTSGTrainingManager(config_gnn, model)
    
    # Train from scratch
    model, trainer = manager.train(train_loader, val_loader)
    
    # Load pretrained checkpoint
    # ckpt_path = os.path.join(config_gnn.save_dir, 'best.ckpt')
    # if os.path.exists(ckpt_path):
    #     model = load_checkpoint(model, ckpt_path, config_gnn.device)
    #     print(f"Loaded checkpoint from {ckpt_path}")
    # else:
    #     print("No checkpoint found, using random initialization")
    # model = model.to(config_gnn.device)

    # print("\n" + "="*60)
    # print("Quick Validation")
    # print("="*60)
    # val_metrics = manager.validate(val_loader, 0)
    # print(f"Validation Accuracy: {val_metrics['acc']:.4f}")

    # print("\n" + "="*60)
    # print("Edge-based OOD Detection")
    # print("="*60)
    
    # metrics = run_edge_ood_detection(model, test_loader, test_ood_loader, config_gnn, loader_dict)
    
    # print("\nDone!")
