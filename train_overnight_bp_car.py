"""
Overnight training script: all 7 model configs × 50 seeds × 100 epochs.

Results saved incrementally to results/overnight/
  - results/overnight/seed_{seed}/{model_name}.npz  per run
  - results/overnight/summary.csv                   final aggregate
"""

import os
import sys
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import mne

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# ── Constants (match notebook) ────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, 'physionet.org/files/eegmmidb/1.0.0')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', 'overnight_excl6_car')
CACHE_PATH  = os.path.join(SCRIPT_DIR, 'results', 'eeg_cache_excl6_car.npz')
os.makedirs(RESULTS_DIR, exist_ok=True)

RUNS       = ['R03', 'R04', 'R07', 'R08', 'R11', 'R12']
N_SUBJECTS = 109
EXCLUDED_SUBJECTS = {38, 82, 88, 89, 100, 104}
SFREQ      = 160
TMIN, TMAX = 0.0, 2.0
N_SAMPLES  = int(SFREQ * (TMAX - TMIN))
N_CHANNELS = 64

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
BATCH_SIZE = 128
EPOCHS     = 100
LR         = 2e-3

DATA_SPLIT_SEED = 35  # fixed — same split every run
N_SEEDS         = 40  # model init / shuffle seeds
SEEDS           = list(range(N_SEEDS))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── Data loading ──────────────────────────────────────────────

def load_subject(subject_id):
    subject = f'S{subject_id:03d}'
    raws = []
    for run in RUNS:
        path = os.path.join(DATA_DIR, subject, f'{subject}{run}.edf')
        if not os.path.exists(path):
            continue
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raws.append(raw)
    if not raws:
        return np.empty((0, N_CHANNELS, 0)), np.empty((0,), dtype=int)
    raw = mne.concatenate_raws(raws)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    events, event_id = mne.events_from_annotations(raw)
    if 'T1' not in event_id or 'T2' not in event_id:
        return np.empty((0, N_CHANNELS, 0)), np.empty((0,), dtype=int)
    epochs = mne.Epochs(raw, events,
                        event_id={'T1': event_id['T1'], 'T2': event_id['T2']},
                        tmin=TMIN, tmax=TMAX,
                        baseline=None, preload=True, verbose=False)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    y = (y == event_id['T2']).astype(int)
    X_fixed = []
    for x in X:
        if x.shape[1] >= N_SAMPLES:
            x = x[:, :N_SAMPLES]
        else:
            x = np.pad(x, ((0, 0), (0, N_SAMPLES - x.shape[1])))
        X_fixed.append(x)
    return np.stack(X_fixed), y


def load_all_subjects():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    bucket = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(load_subject, sid): sid for sid in range(1, N_SUBJECTS + 1)
                   if sid not in EXCLUDED_SUBJECTS}
        for i, future in enumerate(as_completed(futures), 1):
            sid = futures[future]
            bucket[sid] = future.result()
    X_all, y_all = [], []
    for sid in range(1, N_SUBJECTS + 1):
        if sid in EXCLUDED_SUBJECTS:
            continue
        X, y = bucket[sid]
        if len(X) > 0:
            X_all.append(X); y_all.append(y)
    return np.concatenate(X_all), np.concatenate(y_all)


def normalize(X):
    mu  = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2,  keepdims=True) + 1e-6
    X   = (X - mu) / std
    return X[:, np.newaxis, :, :]


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Models ────────────────────────────────────────────────────

class EEGNet(nn.Module):
    def __init__(self, n_channels=64, n_samples=320, n_classes=2,
                 F1=8, D=2, F2=16, dropout=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(F2 * (n_samples // 32), n_classes)

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.classifier(x.flatten(start_dim=1))


class InceptionTemporalBlock(nn.Module):
    def __init__(self, kernel_sizes, F1=8):
        super().__init__()
        n = len(kernel_sizes)
        base = F1 // n
        branch_filters = [base] * n
        branch_filters[-1] += F1 - base * n
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, f, (1, k), padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(f),
            )
            for k, f in zip(kernel_sizes, branch_filters)
        ])
        total = sum(branch_filters)
        self.project = nn.Conv2d(total, F1, (1, 1), bias=False) if total != F1 else nn.Identity()
        self.bn_out = nn.BatchNorm2d(F1)

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        t = min(o.shape[-1] for o in outs)
        x = torch.cat([o[..., :t] for o in outs], dim=1)
        return self.bn_out(self.project(x))


class EEGNetInception(nn.Module):
    def __init__(self, n_channels=64, n_samples=320, n_classes=2,
                 F1=8, D=2, F2=16, dropout=0.25, kernel_sizes=(16, 32, 64)):
        super().__init__()
        self.block1 = InceptionTemporalBlock(kernel_sizes=kernel_sizes, F1=F1)
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(F2 * (n_samples // 32), n_classes)

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.classifier(x.flatten(start_dim=1))


class InceptionSpatialBlock(nn.Module):
    """Inception block with per-branch depthwise spatial conv — collapses channels inside each branch."""
    def __init__(self, kernel_sizes, n_channels, F1=8, D=2, dropout=0.25):
        super().__init__()
        n = len(kernel_sizes)
        base = F1 // n
        branch_filters = [base] * n
        branch_filters[-1] += F1 - base * n
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, f, (1, k), padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(f),
                nn.Conv2d(f, f * D, (n_channels, 1), groups=f, bias=False),
                nn.BatchNorm2d(f * D), nn.ELU(),
            )
            for k, f in zip(kernel_sizes, branch_filters)
        ])
        total = sum(f * D for f in branch_filters)
        out_ch = F1 * D
        self.project = nn.Conv2d(total, out_ch, (1, 1), bias=False) if total != out_ch else nn.Identity()
        self.bn_out = nn.BatchNorm2d(out_ch)
        self.pool = nn.AvgPool2d((1, 4))
        self.drop = nn.Dropout(dropout)
        self.out_ch = out_ch

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        t = min(o.shape[-1] for o in outs)
        x = torch.cat([o[..., :t] for o in outs], dim=1)
        return self.drop(self.pool(self.bn_out(self.project(x))))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
        return x * w


class EEGNetInceptionSpatial(nn.Module):
    """Inception with per-branch spatial filtering (replaces shared depthwise block)."""
    def __init__(self, n_channels=64, n_samples=320, n_classes=2,
                 F1=8, D=2, F2=16, dropout=0.25, kernel_sizes=(16, 32, 64)):
        super().__init__()
        self.block1 = InceptionSpatialBlock(kernel_sizes, n_channels, F1, D, dropout)
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(F2 * (n_samples // 32), n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block3(x)
        return self.classifier(x.flatten(start_dim=1))


class EEGNetInceptionSE(nn.Module):
    """Inception with SE channel attention after temporal multi-scale block."""
    def __init__(self, n_channels=64, n_samples=320, n_classes=2,
                 F1=8, D=2, F2=16, dropout=0.25, kernel_sizes=(16, 32, 64)):
        super().__init__()
        self.block1 = InceptionTemporalBlock(kernel_sizes=kernel_sizes, F1=F1)
        self.se = SEBlock(F1)
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(F2 * (n_samples // 32), n_classes)

    def forward(self, x):
        x = self.se(self.block1(x))
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x.flatten(start_dim=1))


class EEGNetInceptionSpatialSE(nn.Module):
    """Inception with per-branch spatial filtering + SE channel attention."""
    def __init__(self, n_channels=64, n_samples=320, n_classes=2,
                 F1=8, D=2, F2=16, dropout=0.25, kernel_sizes=(16, 32, 64)):
        super().__init__()
        self.block1 = InceptionSpatialBlock(kernel_sizes, n_channels, F1, D, dropout)
        self.se = SEBlock(F1 * D)
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(F2 * (n_samples // 32), n_classes)

    def forward(self, x):
        x = self.se(self.block1(x))
        x = self.block3(x)
        return self.classifier(x.flatten(start_dim=1))


class EEGITNet(nn.Module):
    def __init__(self, n_channels=64, n_samples=320, n_classes=2,
                 n_ff=(4, 8, 16), dropout=0.4):
        super().__init__()
        total_f = sum(n_ff)
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, n_ff[0], (1, 16), padding=(0, 8),  bias=False),
            nn.BatchNorm2d(n_ff[0]), nn.ELU(),
            nn.Conv2d(n_ff[0], n_ff[0], (n_channels, 1), groups=n_ff[0], bias=False),
            nn.BatchNorm2d(n_ff[0]), nn.ELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, n_ff[1], (1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(n_ff[1]), nn.ELU(),
            nn.Conv2d(n_ff[1], n_ff[1], (n_channels, 1), groups=n_ff[1], bias=False),
            nn.BatchNorm2d(n_ff[1]), nn.ELU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, n_ff[2], (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(n_ff[2]), nn.ELU(),
            nn.Conv2d(n_ff[2], n_ff[2], (n_channels, 1), groups=n_ff[2], bias=False),
            nn.BatchNorm2d(n_ff[2]), nn.ELU(),
        )
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop  = nn.Dropout(dropout)
        self.dilations = [1, 2, 4, 8]
        self.tcn = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(total_f, total_f, (1, 4), dilation=(1, d), groups=total_f, bias=False),
                nn.BatchNorm2d(total_f), nn.ELU(), nn.Dropout(dropout),
                nn.Conv2d(total_f, total_f, (1, 4), dilation=(1, d), groups=total_f, bias=False),
                nn.BatchNorm2d(total_f), nn.ELU(), nn.Dropout(dropout),
            ])
            for d in self.dilations
        ])
        self.out_conv = nn.Sequential(
            nn.Conv2d(total_f, total_f, (1, 1)),
            nn.BatchNorm2d(total_f), nn.ELU(),
        )
        self.pool2 = nn.AvgPool2d((1, 4))
        self.classifier = nn.Linear(total_f * (n_samples // 16), n_classes)

    def _tcn_block(self, layers, x, d):
        pad = 3 * d
        res = x
        out = layers[0](F.pad(x, (pad, 0)))
        out = layers[1](out); out = layers[2](out); out = layers[3](out)
        out = layers[4](F.pad(out, (pad, 0)))
        out = layers[5](out); out = layers[6](out); out = layers[7](out)
        return out + res

    def forward(self, x):
        b1, b2, b3 = self.branch1(x), self.branch2(x), self.branch3(x)
        t = min(b1.shape[-1], b2.shape[-1], b3.shape[-1])
        x = torch.cat([b1[..., :t], b2[..., :t], b3[..., :t]], dim=1)
        x = self.drop(self.pool1(x))
        for layers, d in zip(self.tcn, self.dilations):
            x = self._tcn_block(layers, x, d)
        x = self.drop(self.pool2(self.out_conv(x)))
        return self.classifier(x.flatten(start_dim=1))


COMMON = dict(n_channels=N_CHANNELS, n_samples=N_SAMPLES)

MODEL_CONFIGS = {
    # 'EEGNet_Inc2_SmallMed': lambda: EEGNetInception(**COMMON, kernel_sizes=(16, 32)),
    # 'EEGNet_Inc2_MedLarge': lambda: EEGNetInception(**COMMON, kernel_sizes=(32, 64)),
    # 'EEGNet_Inc3':          lambda: EEGNetInception(**COMMON, kernel_sizes=(16, 32, 64)),
    # 'EEGNet_Inc3_Narrow':   lambda: EEGNetInception(**COMMON, kernel_sizes=(8, 16, 32)),
    # 'EEGNet_Inc3_Wide':     lambda: EEGNetInception(**COMMON, kernel_sizes=(32, 64, 128)),
    # Spatial variants — per-branch depthwise spatial conv
    # 'Inc_Spatial_SmallMed': lambda: EEGNetInceptionSpatial(**COMMON, kernel_sizes=(16, 32)),
    # 'Inc_Spatial_MedLarge': lambda: EEGNetInceptionSpatial(**COMMON, kernel_sizes=(32, 64)),
    # 'Inc_Spatial_Inc3':     lambda: EEGNetInceptionSpatial(**COMMON, kernel_sizes=(16, 32, 64)),
    # 'Inc_Spatial_Narrow':   lambda: EEGNetInceptionSpatial(**COMMON, kernel_sizes=(8, 16, 32)),
    # 'Inc_Spatial_Wide':     lambda: EEGNetInceptionSpatial(**COMMON, kernel_sizes=(32, 64, 128)),
    # SE variants — channel attention after temporal inception
    'Inc_SE_SmallMed':      lambda: EEGNetInceptionSE(**COMMON, kernel_sizes=(16, 32)),
    'Inc_SE_MedLarge':      lambda: EEGNetInceptionSE(**COMMON, kernel_sizes=(32, 64)),
    'Inc_SE_Inc3':          lambda: EEGNetInceptionSE(**COMMON, kernel_sizes=(16, 32, 64)),
    'Inc_SE_Narrow':        lambda: EEGNetInceptionSE(**COMMON, kernel_sizes=(8, 16, 32)),
    'Inc_SE_Narrow3':        lambda: EEGNetInceptionSE(**COMMON, F1=4, D=4, F2=16,  kernel_sizes=(8, 16, 32)),
    'Inc_SE_Narrow4':        lambda: EEGNetInceptionSE(**COMMON, F1=8, D=1, F2=32, kernel_sizes=(8, 16, 32)),
    'Inc_SE_Narrow2':        lambda: EEGNetInceptionSE(**COMMON, F1=8, D=4, F2=16, kernel_sizes=(8, 16, 32)),
    'Inc_SE_Narrow5':        lambda: EEGNetInceptionSE(**COMMON, F1=8, D=2, F2=24,  kernel_sizes=(8, 16, 32)),

    'EEGNet':               lambda: EEGNet(**COMMON),
    'EEGITNet':             lambda: EEGITNet(**COMMON, n_ff=(4, 8, 16)),

    # 'Inc_SE_Wide':          lambda: EEGNetInceptionSE(**COMMON, kernel_sizes=(32, 64, 128)),
    # Spatial + SE combined
    # 'Inc_SpatialSE_SmallMed': lambda: EEGNetInceptionSpatialSE(**COMMON, kernel_sizes=(16, 32)),
    # 'Inc_SpatialSE_MedLarge': lambda: EEGNetInceptionSpatialSE(**COMMON, kernel_sizes=(32, 64)),
    # 'Inc_SpatialSE_Inc3':     lambda: EEGNetInceptionSpatialSE(**COMMON, kernel_sizes=(16, 32, 64)),
    # 'Inc_SpatialSE_Narrow':   lambda: EEGNetInceptionSpatialSE(**COMMON, kernel_sizes=(8, 16, 32)),
    # 'Inc_SpatialSE_Wide':     lambda: EEGNetInceptionSpatialSE(**COMMON, kernel_sizes=(32, 64, 128)),
}


# ── Training ──────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_one_run(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.25)
    criterion = nn.CrossEntropyLoss()

    train_losses = np.zeros(epochs)
    train_accs   = np.zeros(epochs)
    val_accs     = np.zeros(epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        tr_correct = tr_total = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total   += len(y)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                correct += (model(X).argmax(1) == y).sum().item()
                total   += len(y)

        train_losses[epoch] = total_loss
        train_accs[epoch]   = tr_correct / tr_total
        val_accs[epoch]     = correct / total
        scheduler.step()

    return train_losses, train_accs, val_accs


def get_split_acc(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            correct += (model(X).argmax(1) == y).sum().item()
            total   += len(y)
    return correct / total


# ── Main ──────────────────────────────────────────────────────

def main():
    # Load / cache data
    if os.path.exists(CACHE_PATH):
        print('Loading data from cache...')
        _c = np.load(CACHE_PATH)
        X_all, y_all = _c['X'], _c['y']
    else:
        print('Loading all subjects (may take a few minutes)...')
        X_all, y_all = load_all_subjects()
        np.savez(CACHE_PATH, X=X_all, y=y_all)
        print('Cached.')

    print(f'Dataset: {X_all.shape} | Classes: {np.bincount(y_all)}')

    X_norm = normalize(X_all)
    n_total = len(X_norm)

    # Fixed data split
    rng = np.random.default_rng(DATA_SPLIT_SEED)
    perm = rng.permutation(n_total)
    n_train = int(n_total * TRAIN_FRAC)
    n_val   = int(n_total * VAL_FRAC)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    full_dataset = EEGDataset(X_norm, y_all)
    val_loader   = DataLoader(Subset(full_dataset, val_idx),  batch_size=BATCH_SIZE)
    test_loader  = DataLoader(Subset(full_dataset, test_idx), batch_size=BATCH_SIZE)

    print(f'Split — train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}')
    print(f'\nModels: {list(MODEL_CONFIGS.keys())}')
    print(f'Seeds: {N_SEEDS} | Epochs: {EPOCHS}')
    print(f'Total runs: {len(MODEL_CONFIGS) * N_SEEDS}\n')

    total_runs = len(MODEL_CONFIGS) * N_SEEDS
    completed  = 0
    t0_global  = time.time()

    summary_rows = []

    for model_name, model_fn in MODEL_CONFIGS.items():
        model_dir = os.path.join(RESULTS_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        model_train_losses = []
        model_train_accs   = []
        model_val_accs     = []
        model_test_accs    = []

        for seed in SEEDS:
            out_path = os.path.join(model_dir, f'seed_{seed:03d}.npz')
            if os.path.exists(out_path):
                # Resume: load existing result
                _r = np.load(out_path)
                model_train_losses.append(_r['train_losses'])
                model_train_accs.append(_r['train_accs'])
                model_val_accs.append(_r['val_accs'])
                model_test_accs.append(float(_r['test_acc']))
                completed += 1
                print(f'[skip] {model_name} seed {seed:3d} — already done')
                continue

            t0 = time.time()
            set_seed(seed)

            # Re-seed the train loader shuffle each seed
            train_loader = DataLoader(
                Subset(full_dataset, train_idx),
                batch_size=BATCH_SIZE, shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )

            model = model_fn()
            train_losses, train_accs, val_accs = train_one_run(model, train_loader, val_loader)
            test_acc = get_split_acc(model, test_loader)

            np.savez(out_path,
                     train_losses=train_losses,
                     train_accs=train_accs,
                     val_accs=val_accs,
                     test_acc=np.float32(test_acc),
                     seed=seed)

            model_train_losses.append(train_losses)
            model_train_accs.append(train_accs)
            model_val_accs.append(val_accs)
            model_test_accs.append(test_acc)

            completed += 1
            elapsed = time.time() - t0
            total_elapsed = time.time() - t0_global
            eta = total_elapsed / completed * (total_runs - completed)
            print(
                f'[{completed:4d}/{total_runs}] {model_name:<25} seed {seed:3d} '
                f'| test_acc={test_acc:.4f} | train_acc={train_accs[-1]:.4f} | val_acc={val_accs[-1]:.4f} '
                f'| {elapsed:.0f}s | ETA {eta/60:.1f}min'
            )
            sys.stdout.flush()

        # Aggregate stats for this model
        arr_losses  = np.array(model_train_losses)  # (N_SEEDS, EPOCHS)
        arr_taccs   = np.array(model_train_accs)    # (N_SEEDS, EPOCHS)
        arr_vaccs   = np.array(model_val_accs)      # (N_SEEDS, EPOCHS)
        arr_tacc    = np.array(model_test_accs)     # (N_SEEDS,)

        np.savez(
            os.path.join(model_dir, 'aggregate.npz'),
            train_losses=arr_losses,
            train_accs=arr_taccs,
            val_accs=arr_vaccs,
            test_accs=arr_tacc,
            seeds=np.array(SEEDS),
        )

        summary_rows.append(dict(
            model=model_name,
            params=sum(p.numel() for p in model_fn().parameters() if p.requires_grad),
            test_acc_mean=arr_tacc.mean(),
            test_acc_std=arr_tacc.std(),
            test_acc_max=arr_tacc.max(),
            final_train_acc_mean=arr_taccs[:, -1].mean(),
            final_train_acc_std=arr_taccs[:, -1].std(),
            final_val_acc_mean=arr_vaccs[:, -1].mean(),
            final_val_acc_std=arr_vaccs[:, -1].std(),
        ))

        print(f'\n  {model_name}: test_acc = {arr_tacc.mean():.4f} ± {arr_tacc.std():.4f}  '
              f'(max {arr_tacc.max():.4f})\n')

    # Write summary CSV
    csv_path = os.path.join(RESULTS_DIR, 'summary.csv')
    with open(csv_path, 'w') as f:
        f.write('model,params,test_acc_mean,test_acc_std,test_acc_max,'
                'final_train_acc_mean,final_train_acc_std,'
                'final_val_acc_mean,final_val_acc_std\n')
        for r in summary_rows:
            f.write(
                f'{r["model"]},{r["params"]},'
                f'{r["test_acc_mean"]:.4f},{r["test_acc_std"]:.4f},{r["test_acc_max"]:.4f},'
                f'{r["final_train_acc_mean"]:.4f},{r["final_train_acc_std"]:.4f},'
                f'{r["final_val_acc_mean"]:.4f},{r["final_val_acc_std"]:.4f}\n'
            )

    total_time = time.time() - t0_global
    print(f'\nDone. Total time: {total_time/3600:.2f}h')
    print(f'Summary → {csv_path}')
    print(f'\nFINAL SUMMARY')
    print(f'{"Model":<25} {"Params":>8} {"Test Acc":>12} {"Max":>8}')
    print('-' * 60)
    for r in summary_rows:
        print(f'{r["model"]:<25} {r["params"]:>8,} '
              f'{r["test_acc_mean"]:.4f}±{r["test_acc_std"]:.4f} '
              f'{r["test_acc_max"]:>8.4f}')


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
    main()
