import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.utils import config as config_utils


ID_CLASS_STYLE = {
    0: ('QAM64', '#ff8c42'),
    1: ('QPSK', '#ffbe0b'),
    2: ('BPSK', '#ff5d8f'),
    3: ('AM-SSB', '#8338ec'),
    4: ('WBFM', '#3a86ff'),
    5: ('CPFSK', '#2ec4b6'),
    6: ('GFSK', '#06d6a0'),
    7: ('PAM4', '#bc5090'),
}

OOD_CLASS_STYLE = {
    'QAM16': ('QAM16', '#c0c0c0'),
    '8PSK': ('8PSK', '#7a7a7a'),
    'AM-DSB': ('AM-DSB', '#000000'),
}


DEFAULT_CONFIGS = [
    'configs/datasets/fine-grained/rml201610a.yml',
    'configs/datasets/fine-grained/rml201610a_ood.yml',
    'configs/networks/resnet18_32x32.yml',
    'configs/pipelines/test/test_ood.yml',
    'configs/preprocessors/base_preprocessor.yml',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='t-SNE feature visualization for fine-grained dataset')
    parser.add_argument('--checkpoint',
                        type=str,
                        default='results_vos/best2.ckpt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output',
                        type=str,
                        default='results_fine_grained/feature_tsne.png',
                        help='Figure save path')
    parser.add_argument('--config',
                        nargs='*',
                        default=DEFAULT_CONFIGS,
                        help='Config files, override defaults if needed')
    parser.add_argument('--id-samples',
                        type=int,
                        default=2000,
                        help='Max number of ID samples for t-SNE')
    parser.add_argument('--ood-samples',
                        type=int,
                        default=2000,
                        help='Max number of OOD samples for t-SNE')
    parser.add_argument('--perplexity',
                        type=float,
                        default=35.0,
                        help='TSNE perplexity')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--ood-split',
                        type=str,
                        default='val',
                        choices=['val', 'nearood', 'farood'],
                        help='Which OOD split to draw from')
    parser.add_argument('--ood-classes',
                        type=int,
                        default=3,
                        help='Expected number of OOD classes to cover')
    parser.add_argument('--plot-logits',
                        action='store_true',
                        help='Additionally plot ID/OOD logit distributions')
    parser.add_argument('--logit-output',
                        type=str,
                        default='results_fine_grained/logit_hist.png',
                        help='Figure path for logit distribution plot')
    parser.add_argument('--logit-bins',
                        type=int,
                        default=60,
                        help='Histogram bins for logit plots')
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_modulation(name: str):
    if name is None:
        return ''
    parts = name.replace('\\', '/').split('/')
    return parts[1] if len(parts) > 1 else name


def build_loaders(cfg, ood_split: str):
    loader_dict = get_dataloader(cfg)
    ood_loader_dict = get_ood_dataloader(cfg)
    id_loader = loader_dict.get('test') or loader_dict['val']
    ood_loader = ood_loader_dict[ood_split]
    return id_loader, ood_loader


def load_model(cfg, checkpoint: str, device: torch.device):
    num_classes = cfg.dataset.num_classes
    model = ResNet18_32x32(num_classes=num_classes).to(device)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get('state_dict', state)
    cleaned_state = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        cleaned_state[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f'[Warn] Missing keys: {missing}')
    if unexpected:
        print(f'[Warn] Unexpected keys: {unexpected}')
    model.eval()
    return model


def _balanced_indices(labels: torch.Tensor, max_samples: int):
    total = labels.size(0)
    if total <= max_samples:
        return torch.arange(total)
    unique = labels.unique()
    if max_samples < len(unique):
        print('[Warn] max_samples smaller than class count; falling back to '
              'random sampling.')
        return torch.randperm(total)[:max_samples]
    selected = []
    chosen_mask = torch.zeros(total, dtype=torch.bool)
    for cls in unique:
        cls_idx = torch.nonzero(labels == cls, as_tuple=False).squeeze(1)
        perm = cls_idx[torch.randperm(len(cls_idx))]
        chosen = perm[:1]
        selected.append(chosen)
        chosen_mask[chosen] = True
    remaining = torch.nonzero(~chosen_mask, as_tuple=False).squeeze(1)
    remaining = remaining[torch.randperm(len(remaining))]
    need = max_samples - len(unique)
    selected.append(remaining[:need])
    return torch.cat(selected)


def collect_features(loader,
                     model,
                     device,
                     max_samples: int,
                     desc: str,
                     label_override=None,
                     min_classes=None,
                     collect_logits: bool = False):
    feats, labels, names = [], [], []
    logits_accum = [] if collect_logits else None
    collected = 0
    seen_classes = set()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            data = batch['data'].to(device)
            logits, feature = model(data, return_feature=True)
            feats.append(feature.cpu())
            if collect_logits:
                logits_accum.append(logits.cpu())
            if label_override is None:
                batch_labels = batch['label'].cpu()
            else:
                batch_labels = torch.full((data.size(0), ),
                                          label_override,
                                          dtype=torch.long)
            labels.append(batch_labels)
            seen_classes.update(batch_labels.unique().tolist())
            batch_names = batch.get('image_name')
            if batch_names is None:
                names.extend([''] * data.size(0))
            elif isinstance(batch_names, (list, tuple)):
                names.extend(batch_names)
            else:
                names.append(batch_names)
            collected += data.size(0)
            if collected >= max_samples and (
                    min_classes is None
                    or len(seen_classes) >= min_classes):
                break

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    names = np.array(names)
    logits_tensor = torch.cat(logits_accum,
                              dim=0) if collect_logits else None
    if min_classes is not None and len(seen_classes) < min_classes:
        print(f'[Warn] Only covered {len(seen_classes)}/{min_classes} classes '
              f'in {desc} loader. Consider increasing --{desc.lower()}-samples.')
    if feats.size(0) > max_samples:
        if label_override is None:
            idx = _balanced_indices(labels, max_samples)
        else:
            idx = torch.randperm(feats.size(0))[:max_samples]
        feats = feats[idx]
        labels = labels[idx]
        names = names[idx.cpu().numpy()]
        if logits_tensor is not None:
            logits_tensor = logits_tensor[idx]
    logits_np = logits_tensor.numpy() if logits_tensor is not None else None
    return feats.numpy(), labels.numpy(), names, logits_np


def run_tsne(id_feats, ood_feats, perplexity, seed):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
    combined = np.concatenate([id_feats, ood_feats], axis=0)
    embedding = tsne.fit_transform(combined)
    id_embed = embedding[:len(id_feats)]
    ood_embed = embedding[len(id_feats):]
    return id_embed, ood_embed


def plot_embedding(id_embed,
                   id_labels,
                   ood_embed,
                   ood_labels,
                   ood_names,
                   output,
                   num_classes: int):
    Path(os.path.dirname(output) or '.').mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab10', num_classes)
    plotted_id = set()
    for cls in range(num_classes):
        mask = id_labels == cls
        if not np.any(mask):
            continue
        name, color = ID_CLASS_STYLE.get(cls,
                                         (f'Class-{cls}',
                                          cmap(cls % cmap.N)))
        plotted_id.add(cls)
        plt.scatter(id_embed[mask, 0],
                    id_embed[mask, 1],
                    s=12,
                    alpha=0.7,
                    color=color,
                    label=f'ID-{name}')
    ood_mods = np.array([extract_modulation(name) for name in ood_names])
    plotted_mods = set()
    for mod_name, (display_name, color) in OOD_CLASS_STYLE.items():
        mask = ood_mods == mod_name
        if np.any(mask):
            plotted_mods.add(mod_name)
            plt.scatter(ood_embed[mask, 0],
                        ood_embed[mask, 1],
                        s=20,
                        alpha=0.85,
                        marker='x',
                        color=color,
                        label=f'OOD-{display_name}')
    remaining_mask = np.isin(ood_mods,
                             list(OOD_CLASS_STYLE.keys()),
                             invert=True)
    if np.any(remaining_mask):
        plt.scatter(ood_embed[remaining_mask, 0],
                    ood_embed[remaining_mask, 1],
                    s=20,
                    alpha=0.85,
                    marker='x',
                    color='#4f4f4f',
                    label='OOD-Other')
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.title('Fine-grained feature t-SNE')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()


def plot_logit_distribution(id_logits, ood_logits, output, bins: int = 60):
    if id_logits is None or ood_logits is None:
        print('[Warn] Logit arrays missing; skip logit plot.')
        return
    Path(os.path.dirname(output) or '.').mkdir(parents=True, exist_ok=True)
    id_flat = id_logits.flatten()
    ood_flat = ood_logits.flatten()
    id_max = id_logits.max(axis=1)
    ood_max = ood_logits.max(axis=1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(id_flat,
             bins=bins,
             alpha=0.6,
             density=True,
             label='ID',
             color='#3a86ff')
    plt.hist(ood_flat,
             bins=bins,
             alpha=0.6,
             density=True,
             label='OOD',
             color='#ff5d8f')
    plt.title('All logit values')
    plt.xlabel('Logit')
    plt.ylabel('Density')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(id_max,
             bins=bins,
             alpha=0.6,
             density=True,
             label='ID max',
             color='#06d6a0')
    plt.hist(ood_max,
             bins=bins,
             alpha=0.6,
             density=True,
             label='OOD max',
             color='#bc5090')
    plt.title('Per-sample max logit')
    plt.xlabel('Max logit')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = config_utils.Config(*args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    id_loader, ood_loader = build_loaders(cfg, args.ood_split)
    model = load_model(cfg, args.checkpoint, device)

    id_feats, id_labels, _, id_logits = collect_features(
        id_loader,
        model,
        device,
        args.id_samples,
        desc='ID',
        min_classes=cfg.dataset.num_classes,
        collect_logits=args.plot_logits)
    ood_feats, ood_labels, ood_names, ood_logits = collect_features(
        ood_loader,
        model,
        device,
        args.ood_samples,
        desc='OOD',
        min_classes=args.ood_classes,
        collect_logits=args.plot_logits)
    # id_embed, ood_embed = run_tsne(id_feats, ood_feats, args.perplexity,
    #                                args.seed)
    # plot_embedding(id_embed, id_labels, ood_embed, ood_labels, ood_names,
    #                args.output, cfg.dataset.num_classes)
    print(f't-SNE figure saved to {args.output}')
    if args.plot_logits:
        plot_logit_distribution(id_logits, ood_logits, args.logit_output,
                                args.logit_bins)
        print(f'Logit distribution figure saved to {args.logit_output}')


if __name__ == '__main__':
    main()

