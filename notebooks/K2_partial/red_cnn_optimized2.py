# -*- coding: utf-8 -*-
"""
RED-CNN untuk Sparse-View Tomographic Image Enhancement
========================================================

Versi Optimasi — multi-core CPU + GPU acceleration:
  - AMP (Automatic Mixed Precision) → training 1.5–2× lebih cepat di GTX 1660 Ti
  - torch.compile (opsional, PyTorch ≥ 2.0) → fusi kernel GPU
  - cuDNN benchmark mode → auto-pilih algoritma konvolusi tercepat
  - DataLoader: num_workers=CPU_COUNT, persistent_workers, prefetch_factor
  - Inference: ThreadPoolExecutor paralel per slice (CPU-bound post-processing)
  - Petrophysics: ProcessPoolExecutor paralel per slice (CPU multiprocessing)
  - ResourceTracker: daemon thread → log CPU%, GPU%, VRAM, RAM, threads setiap N detik

Implementasi berdasarkan:
  Chen et al., "Low-Dose CT with a Residual Encoder-Decoder
  Convolutional Neural Network (RED-CNN)", IEEE TMI 2017.

Dependensi:
  pip install torch torchvision tifffile scikit-image scipy
              matplotlib pandas tqdm psutil gputil
"""

# ============================================================================
# CELL 1: IMPORT
# ============================================================================

import os
import gc
import time
import json
import warnings
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast   # AMP

from tifffile import imread, imwrite
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion
import psutil

warnings.filterwarnings('ignore')

# ── Jumlah CPU core fisik tersedia ───────────────────────────────────────────
CPU_COUNT = multiprocessing.cpu_count()

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    # cuDNN benchmark: auto-pilih algoritma konvolusi tercepat untuk input size tetap
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ============================================================================
# RESOURCE TRACKER  — daemon thread, log setiap INTERVAL detik
# ============================================================================

class ResourceTracker:
    def __init__(self, interval: float = 2.0, log_path: str = "resource_log.csv"):
        self.interval  = interval
        self.log_path  = log_path
        self._stop_evt = threading.Event()
        self._lock     = threading.Lock()
        self._phase    = "idle"
        self._records  = []
        self._thread   = threading.Thread(target=self._run, daemon=True, name="ResourceTracker")

        self._nvml_ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
            self._nvml_ok = True
            print("ResourceTracker: GPU monitoring via pynvml ✓")
        except Exception:
            try:
                import GPUtil
                self._gputil = GPUtil
                print("ResourceTracker: GPU monitoring via GPUtil ✓")
            except Exception:
                print("ResourceTracker: GPU monitoring tidak tersedia (install pynvml atau gputil)")

    def start(self):
        self._thread.start()
        print(f"ResourceTracker: started (interval={self.interval}s, log={self.log_path})")

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=5)
        self._flush_csv()
        print(f"ResourceTracker: stopped — {len(self._records)} records saved → {self.log_path}")

    def set_phase(self, phase: str):
        with self._lock:
            self._phase = phase

    @contextmanager
    def phase(self, name: str):
        self.set_phase(name)
        print(f"\n[ResourceTracker] ── Phase: {name} ──")
        try:
            yield
        finally:
            self.set_phase("idle")

    def _gpu_stats(self):
        gpu_util = 0.0
        vram_used_mb = 0.0
        vram_total_mb = 0.0
        if self._nvml_ok:
            try:
                mem  = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util      = float(util.gpu)
                vram_used_mb  = mem.used  / 1e6
                vram_total_mb = mem.total / 1e6
            except Exception:
                pass
        elif hasattr(self, '_gputil'):
            try:
                gpus = self._gputil.getGPUs()
                if gpus:
                    gpu_util      = gpus[0].load * 100
                    vram_used_mb  = gpus[0].memoryUsed
                    vram_total_mb = gpus[0].memoryTotal
            except Exception:
                pass
        elif device.type == 'cuda':
            vram_used_mb  = torch.cuda.memory_allocated() / 1e6
            vram_total_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
        return gpu_util, vram_used_mb, vram_total_mb

    def _run(self):
        while not self._stop_evt.is_set():
            ts         = datetime.now().isoformat(timespec='seconds')
            cpu_total  = psutil.cpu_percent(interval=None)
            cpu_cores  = psutil.cpu_percent(interval=None, percpu=True)
            ram        = psutil.virtual_memory()
            n_threads  = threading.active_count()
            gpu_util, vram_used, vram_total = self._gpu_stats()

            with self._lock:
                phase = self._phase
                rec = {
                    'timestamp'   : ts,
                    'phase'       : phase,
                    'cpu_total_%' : cpu_total,
                    'ram_used_mb' : ram.used / 1e6,
                    'ram_total_mb': ram.total / 1e6,
                    'ram_%'       : ram.percent,
                    'gpu_util_%'  : gpu_util,
                    'vram_used_mb': vram_used,
                    'vram_total_mb': vram_total,
                    'active_threads': n_threads,
                    'cpu_cores_json': json.dumps(cpu_cores),
                }
                self._records.append(rec)

            self._stop_evt.wait(self.interval)

    def _flush_csv(self):
        if not self._records:
            return
        df = pd.DataFrame(self._records)
        df.to_csv(self.log_path, index=False)

    def plot(self, save_path: str = "resource_usage.png"):
        if not self._records:
            print("ResourceTracker: tidak ada data untuk diplot.")
            return

        df = pd.DataFrame(self._records)
        df['t'] = range(len(df))
        phase_colors = {
            'idle'        : 'lightgray',
            'training'    : 'royalblue',
            'inference'   : 'darkorange',
            'petrophysics': 'forestgreen',
        }

        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
        fig.suptitle('Resource Usage — RED-CNN Pipeline', fontsize=13, fontweight='bold')

        def shade_phases(ax):
            prev_phase, prev_t = df['phase'].iloc[0], 0
            for i, row in df.iterrows():
                if row['phase'] != prev_phase or i == len(df) - 1:
                    clr = phase_colors.get(prev_phase, 'lightyellow')
                    ax.axvspan(prev_t, row['t'], alpha=0.15, color=clr, label=prev_phase)
                    prev_phase = row['phase']
                    prev_t = row['t']

        axes[0].plot(df['t'], df['cpu_total_%'], color='steelblue', lw=1.5, label='CPU Total %')
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel('CPU %'); axes[0].set_title('CPU Utilization')
        axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
        shade_phases(axes[0])

        axes[1].plot(df['t'], df['ram_used_mb'] / 1024, color='purple', lw=1.5, label='RAM Used (GB)')
        axes[1].set_ylabel('GB'); axes[1].set_title('RAM Usage')
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
        shade_phases(axes[1])

        axes[2].plot(df['t'], df['gpu_util_%'], color='tomato', lw=1.5, label='GPU Util %')
        axes[2].set_ylim(0, 105)
        axes[2].set_ylabel('GPU %'); axes[2].set_title('GPU Utilization')
        axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
        shade_phases(axes[2])

        axes[3].plot(df['t'], df['vram_used_mb'] / 1024, color='darkorange', lw=1.5, label='VRAM Used (GB)')
        axes[3].set_ylabel('GB'); axes[3].set_title('VRAM Usage')
        axes[3].set_xlabel(f'Sample (interval={self.interval}s)')
        axes[3].legend(fontsize=8); axes[3].grid(True, alpha=0.3)
        shade_phases(axes[3])

        from matplotlib.patches import Patch
        handles = [Patch(facecolor=c, alpha=0.4, label=p)
                   for p, c in phase_colors.items()]
        fig.legend(handles=handles, loc='upper right', ncol=4,
                   fontsize=8, title='Phase', title_fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Resource plot saved: {save_path}")

    def summary(self):
        if not self._records:
            return
        df = pd.DataFrame(self._records)
        print("\n" + "="*65)
        print("  RESOURCE SUMMARY PER PHASE")
        print("="*65)
        for phase, grp in df.groupby('phase'):
            print(f"\n  [{phase.upper()}]")
            print(f"    Duration       : {len(grp) * self.interval:.0f}s  ({len(grp)} samples)")
            print(f"    CPU avg/max    : {grp['cpu_total_%'].mean():.1f}% / {grp['cpu_total_%'].max():.1f}%")
            print(f"    RAM avg/max    : {grp['ram_used_mb'].mean()/1024:.2f} / {grp['ram_used_mb'].max()/1024:.2f} GB")
            print(f"    GPU avg/max    : {grp['gpu_util_%'].mean():.1f}% / {grp['gpu_util_%'].max():.1f}%")
            print(f"    VRAM avg/max   : {grp['vram_used_mb'].mean()/1024:.2f} / {grp['vram_used_mb'].max()/1024:.2f} GB")
            print(f"    Active threads : {grp['active_threads'].mean():.1f} avg")


# ============================================================================
# CELL 2: KONFIGURASI
# ============================================================================

class Config:
    PATH_02 = r'D:/Alfian_TA/RED_CNN/Training Data/Libo 512/ground_truth_crop_z200-712_y200-712_x200-712.tif'
    PATH_04 = r'D:/Alfian_TA/RED_CNN/Training Data/Libo 512/poor_04_crop_z200-712_y200-712_x200-712.tif'
    PATH_08 = r'D:/Alfian_TA/RED_CNN/Training Data/Libo 512/poor_08_crop_z200-712_y200-712_x200-712.tif'

    TRAINING_PAIR = 'both'

    CROP_SIZE   = None
    CROP_ORIGIN = None

    TRAIN_FRAC = 0.80
    VAL_FRAC   = 0.10

    PATCH_SIZE        = 256
    PATCHES_PER_SLICE = 20
    STRIDE_TEST       = 96

    BATCH_SIZE  = 32
    EPOCHS      = 100
    LR          = 1e-4
    LR_MIN      = 1e-6
    PATIENCE_ES = 20
    PATIENCE_LR = 8
    LR_FACTOR   = 0.5

    LOSS_TYPE   = 'mse+ssim'
    SSIM_WEIGHT = 0.2

    AUGMENT = True

    VOXEL_SIZE_UM   = 5.714
    KC_TORTUOSITY   = 2.5
    KC_SHAPE_FACTOR = 2.0

    SAVE_DIR   = r'D:/Alfian_TA/RED_CNN'
    MODEL_NAME = 'red_cnn_best.pth'

    USE_AMP = False
    USE_COMPILE = False
    NUM_WORKERS = 4
    TRACKER_INTERVAL = 2.0

cfg = Config()

# ============================================================================
# CELL 3: ARSITEKTUR RED-CNN
# ============================================================================

class REDCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_filters: int = 96,
                 kernel_size: int = 5, num_layers: int = 5):
        super().__init__()
        self.num_layers = num_layers
        pad = kernel_size // 2

        enc_layers = [nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size, 1, pad),
            nn.ReLU(inplace=True)
        )]
        for _ in range(num_layers - 1):
            enc_layers.append(nn.Sequential(
                nn.Conv2d(num_filters, num_filters, kernel_size, 1, pad),
                nn.ReLU(inplace=True)
            ))
        self.encoders = nn.ModuleList(enc_layers)

        dec_layers = []
        for _ in range(num_layers - 1):
            dec_layers.append(nn.Sequential(
                nn.ConvTranspose2d(num_filters, num_filters, kernel_size, 1, pad),
                nn.ReLU(inplace=True)
            ))
        dec_layers.append(
            nn.ConvTranspose2d(num_filters, in_channels, kernel_size, 1, pad)
        )
        self.decoders = nn.ModuleList(dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        enc_features = []
        for enc in self.encoders:
            x = enc(x)
            enc_features.append(x)
        for i, dec in enumerate(self.decoders):
            sc = self.num_layers - 2 - i
            x  = dec(x)
            if sc >= 0:
                x = torch.relu(x + enc_features[sc])
        return x + input_x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# CELL 4: DATASET & DATA LOADING CLASSES
# ============================================================================

def load_and_crop_volume(tif_path: str, crop_size, crop_origin) -> np.ndarray:
    print(f"  Loading: {os.path.basename(tif_path)} ...", end='', flush=True)
    vol = imread(tif_path).astype(np.float32)
    print(f" shape={vol.shape}")
    if crop_size is not None:
        D, H, W = vol.shape
        cs = crop_size
        if crop_origin is not None:
            x0, y0, z0 = crop_origin
        else:
            x0 = max(0, (W-cs)//2); y0 = max(0, (H-cs)//2); z0 = max(0, (D-cs)//2)
        x0 = min(x0, W-cs); y0 = min(y0, H-cs); z0 = min(z0, D-cs)
        vol = vol[z0:z0+cs, y0:y0+cs, x0:x0+cs]
        print(f"  → Cropped → {vol.shape}")
    return vol


def normalize_volume(vol: np.ndarray, p1=None, p99=None) -> tuple:
    if p1  is None: p1  = np.percentile(vol, 1)
    if p99 is None: p99 = np.percentile(vol, 99)
    vol_n = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    return vol_n.astype(np.float32), p1, p99


def split_slices(n_slices: int, train_frac: float, val_frac: float):
    n_train = int(n_slices * train_frac)
    n_val   = int(n_slices * val_frac)
    return (list(range(0, n_train)),
            list(range(n_train, n_train + n_val)),
            list(range(n_train + n_val, n_slices)))


class PatchDataset(Dataset):
    def __init__(self, noisy_vol, clean_vol, slice_indices, patch_size,
                 patches_per_slice=20, augment=False, mode='train'):
        self.noisy  = noisy_vol
        self.clean  = clean_vol
        self.patch  = patch_size
        self.augment = augment
        self.mode   = mode
        self.patches_per_slice = patches_per_slice

        if mode == 'train':
            self.slice_indices = slice_indices
            self.length = len(slice_indices) * patches_per_slice
        else:
            stride = max(patch_size // 2, 32)
            self.coords = []
            _, H, W = noisy_vol.shape
            for z in slice_indices:
                for y in range(0, H - patch_size + 1, stride):
                    for x in range(0, W - patch_size + 1, stride):
                        self.coords.append((z, y, x))
            self.length = len(self.coords)

    def __len__(self):
        return self.length

    def _augment(self, n, c):
        k = np.random.randint(0, 4)
        n = np.rot90(n, k); c = np.rot90(c, k)
        if np.random.rand() > 0.5: n = np.fliplr(n); c = np.fliplr(c)
        if np.random.rand() > 0.5: n = np.flipud(n); c = np.flipud(c)
        return n.copy(), c.copy()

    def __getitem__(self, idx):
        ps = self.patch
        _, H, W = self.noisy.shape
        if self.mode == 'train':
            slice_idx = self.slice_indices[idx // self.patches_per_slice]
            y = np.random.randint(0, H - ps)
            x = np.random.randint(0, W - ps)
        else:
            z, y, x = self.coords[idx]
            slice_idx = z

        n_p = self.noisy[slice_idx, y:y+ps, x:x+ps]
        c_p = self.clean[slice_idx, y:y+ps, x:x+ps]

        if self.augment and self.mode == 'train':
            n_p, c_p = self._augment(n_p, c_p)

        return (torch.from_numpy(n_p[None]).float(),
                torch.from_numpy(c_p[None]).float())


# ============================================================================
# CELL 5: LOSS FUNCTION
# ============================================================================

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.ws = window_size; self.C1 = C1; self.C2 = C2
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / (2*sigma**2))
            for x in range(window_size)
        ])
        gauss /= gauss.sum()
        k = gauss.unsqueeze(1).mm(gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', k)

    def forward(self, pred, target):
        p = self.ws // 2
        k = self.kernel.expand(pred.shape[1], 1, -1, -1)
        F = nn.functional.conv2d
        mu_p  = F(pred,   k, padding=p, groups=1)
        mu_t  = F(target, k, padding=p, groups=1)
        mu_pp = mu_p*mu_p; mu_tt = mu_t*mu_t; mu_pt = mu_p*mu_t
        s_pp  = F(pred*pred,     k, padding=p, groups=1) - mu_pp
        s_tt  = F(target*target, k, padding=p, groups=1) - mu_tt
        s_pt  = F(pred*target,   k, padding=p, groups=1) - mu_pt
        ssim  = ((2*mu_pt+self.C1)*(2*s_pt+self.C2)) / \
                ((mu_pp+mu_tt+self.C1)*(s_pp+s_tt+self.C2))
        return 1.0 - ssim.mean()


class CombinedLoss(nn.Module):
    def __init__(self, ssim_weight=0.2):
        super().__init__()
        self.mse  = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.w    = ssim_weight

    def forward(self, pred, target):
        return (1 - self.w)*self.mse(pred, target) + self.w*self.ssim(pred, target)


# ============================================================================
# CELL 6: TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.USE_AMP):
            loss = criterion(model(noisy), clean)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        with autocast(enabled=cfg.USE_AMP):
            total += criterion(model(noisy), clean).item()
    return total / len(loader)


def train_model(cfg, tracker: ResourceTracker, train_loader, val_loader, norm_stats):
    model = REDCNN(in_channels=1, num_filters=96, kernel_size=5).to(device)
    print(f"  Model parameters: {count_parameters(model):,}")

    if cfg.USE_COMPILE:
        try:
            model = torch.compile(model)
            print("  torch.compile: ✓ (pertama kali akan ada overhead kompilasi)")
        except Exception as e:
            print(f"  torch.compile: gagal ({e}), lanjut tanpa compile")

    criterion = (CombinedLoss(cfg.SSIM_WEIGHT) if cfg.LOSS_TYPE == 'mse+ssim'
                 else nn.MSELoss()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = ReduceLROnPlateau(optimizer, 'min', cfg.LR_FACTOR,
                                  cfg.PATIENCE_LR, min_lr=cfg.LR_MIN)

    scaler = GradScaler(enabled=cfg.USE_AMP)

    best_val = float('inf')
    pat_cnt  = 0
    history  = {'train_loss': [], 'val_loss': [], 'lr': []}
    model_path = os.path.join(cfg.SAVE_DIR, cfg.MODEL_NAME)

    print(f"\n  Training started : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  AMP              : {cfg.USE_AMP}")
    print(f"  Save path        : {model_path}\n")
    t_start = time.time()

    with tracker.phase("training"):
        for epoch in range(1, cfg.EPOCHS + 1):
            t_ep    = time.time()
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            vl_loss = validate(model, val_loader, criterion, device)
            scheduler.step(vl_loss)

            lr_now = optimizer.param_groups[0]['lr']
            history['train_loss'].append(tr_loss)
            history['val_loss'].append(vl_loss)
            history['lr'].append(lr_now)

            if vl_loss < best_val:
                best_val = vl_loss; pat_cnt = 0
                torch.save({
                    'epoch'      : epoch,
                    'model_state': model.state_dict(),
                    'optimizer'  : optimizer.state_dict(),
                    'val_loss'   : best_val,
                    'config'     : {k: v for k, v in cfg.__dict__.items()
                                    if not k.startswith('__')},
                    'norm_stats' : norm_stats,
                }, model_path)
                marker = " ← best"
            else:
                pat_cnt += 1
                marker = f" (patience {pat_cnt}/{cfg.PATIENCE_ES})"

            if epoch % 5 == 0 or epoch == 1:
                elapsed = (time.time() - t_start) / 60
                ep_time = time.time() - t_ep
                avg_ep  = elapsed / epoch * 60
                remain  = avg_ep * (cfg.EPOCHS - epoch) / 60
                print(f"  Ep {epoch:4d}/{cfg.EPOCHS}  "
                      f"tr={tr_loss:.5f}  vl={vl_loss:.5f}  "
                      f"lr={lr_now:.2e}  {ep_time:.1f}s/ep  "
                      f"elapsed={elapsed:.1f}min  ETA≈{remain:.1f}min{marker}")

            if pat_cnt >= cfg.PATIENCE_ES:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    total_time = (time.time() - t_start) / 60
    print(f"\n  Training finished : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Total time        : {total_time:.1f} min")
    print(f"  Best val loss     : {best_val:.6f}")
    return model, history, model_path


# ============================================================================
# CELL 7: PLOT TRAINING HISTORY
# ============================================================================

def plot_training_history(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='royalblue')
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss',   color='tomato')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(epochs, history['lr'], color='forestgreen')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Learning Rate (log scale)')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, 'training_history.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p}")


# ============================================================================
# CELL 8: INFERENCE
# ============================================================================

@torch.no_grad()
def _inference_single_slice(args):
    model, noisy_slice, device, patch_size, stride = args
    model.eval()
    H, W = noisy_slice.shape
    out  = np.zeros((H, W), np.float64)
    wgt  = np.zeros((H, W), np.float64)
    g    = np.outer(np.hanning(patch_size), np.hanning(patch_size)) + 1e-6

    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))
    if not ys or ys[-1] + patch_size < H: ys.append(max(0, H - patch_size))
    if not xs or xs[-1] + patch_size < W: xs.append(max(0, W - patch_size))

    for y in ys:
        for x in xs:
            t = torch.from_numpy(
                noisy_slice[y:y+patch_size, x:x+patch_size][None, None]
            ).float().to(device)
            with autocast(enabled=cfg.USE_AMP):
                p = model(t).squeeze().cpu().numpy()
            out[y:y+patch_size, x:x+patch_size] += p * g
            wgt[y:y+patch_size, x:x+patch_size] += g

    return np.clip(out / wgt, 0, 1).astype(np.float32)


def run_inference_testset(model, vol_noisy, vol_clean, test_idx, cfg, device, tracker):
    print(f"\n  Running inference on {len(test_idx)} test slices ...")
    model.eval()

    denoised_slices = [None] * len(test_idx)
    psnr_list = []; ssim_list = []; rmse_list = []
    n_workers = min(4, CPU_COUNT)

    with tracker.phase("inference"):
        args_list = [
            (model, vol_noisy[z], device, cfg.PATCH_SIZE, cfg.STRIDE_TEST)
            for z in test_idx
        ]

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_inference_single_slice, args): i
                       for i, args in enumerate(args_list)}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc='  Test inference'):
                i = futures[fut]
                d_sl = fut.result()
                c_sl = vol_clean[test_idx[i]]
                denoised_slices[i] = d_sl
                psnr_list.append((i, psnr_fn(c_sl, d_sl, data_range=1.0)))
                ssim_list.append((i, ssim_fn(c_sl, d_sl, data_range=1.0)))
                rmse_list.append((i, float(np.sqrt(np.mean((c_sl - d_sl)**2)))))

    psnr_list = [v for _, v in sorted(psnr_list)]
    ssim_list = [v for _, v in sorted(ssim_list)]
    rmse_list = [v for _, v in sorted(rmse_list)]

    denoised_vol = np.stack(denoised_slices, axis=0)
    metrics = {
        'PSNR_mean': np.mean(psnr_list), 'PSNR_std': np.std(psnr_list),
        'SSIM_mean': np.mean(ssim_list), 'SSIM_std': np.std(ssim_list),
        'RMSE_mean': np.mean(rmse_list), 'RMSE_std': np.std(rmse_list),
    }

    print(f"\n  ── Test Set Metrics ──────────────────────────────────────────")
    print(f"  PSNR : {metrics['PSNR_mean']:.4f} ± {metrics['PSNR_std']:.4f} dB")
    print(f"  SSIM : {metrics['SSIM_mean']:.4f} ± {metrics['SSIM_std']:.4f}")
    print(f"  RMSE : {metrics['RMSE_mean']:.6f} ± {metrics['RMSE_std']:.6f}")
    return denoised_vol, metrics, psnr_list, ssim_list, rmse_list


# ============================================================================
# CELL 9: PETROPHYSICS — PROCESSPOOL EXECUTOR
# ============================================================================

def binarize_volume(vol, method='otsu'):
    flat   = vol.flatten()
    thresh = threshold_otsu(flat) if method == 'otsu' else np.percentile(flat, 30)
    return (vol < thresh).astype(np.uint8), float(thresh)


def calculate_porosity(binary_vol):
    per_sl = binary_vol.mean(axis=(1, 2))
    return {
        'porosity_total'    : float(binary_vol.mean()),
        'porosity_percent'  : float(binary_vol.mean() * 100),
        'pore_voxels'       : int(binary_vol.sum()),
        'total_voxels'      : int(binary_vol.size),
        'porosity_per_slice': per_sl,
    }


def calculate_dice_iou(pred_bin, true_bin):
    p = pred_bin.astype(bool); r = true_bin.astype(bool)
    tp = np.logical_and(p,  r).sum()
    fp = np.logical_and(p, ~r).sum()
    fn = np.logical_and(~p, r).sum()
    return {'Dice': float(2*tp/(2*tp+fp+fn+1e-8)),
            'IoU' : float(tp/(tp+fp+fn+1e-8)),
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn)}


def _process_single_slice(args):
    z_idx, bin_sl, phi_frac, voxel_um, tortuosity, shape_factor = args
    vox_m = voxel_um * 1e-6
    H, W  = bin_sl.shape

    # SSA
    surf   = bin_sl.astype(int) - binary_erosion(bin_sl).astype(int)
    sa_m2  = surf.sum() * vox_m**2
    vol_m3 = H * W * vox_m**3
    Sv_m   = sa_m2 / vol_m3 if vol_m3 > 0 else 0.0

    # Kozeny-Carman
    if phi_frac > 0 and phi_frac < 1 and Sv_m > 0:
        num  = phi_frac**3
        den  = shape_factor * tortuosity**2 * Sv_m**2 * (1 - phi_frac)**2
        K_m2 = num / den
        K_mD = K_m2 / 9.869233e-16
    else:
        K_mD = float('nan')

    return z_idx, Sv_m, K_mD


def run_petrophysics_parallel(bin_vol, por, voxel_um, tortuosity, shape_factor, label, tracker):
    n = bin_vol.shape[0]
    phi_per_slice = por['porosity_per_slice']
    args_list = [
        (z, bin_vol[z], float(phi_per_slice[z]), voxel_um, tortuosity, shape_factor)
        for z in range(n)
    ]

    Sv_arr   = np.zeros(n)
    K_mD_arr = np.full(n, float('nan'))

    n_workers = max(1, CPU_COUNT - 1)
    print(f"  [{label}] ProcessPoolExecutor: {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_single_slice, args): args[0]
                   for args in args_list}
        for fut in tqdm(as_completed(futures), total=n,
                        desc=f'  Petrophysics [{label}]'):
            z_idx, Sv_m, K_mD = fut.result()
            Sv_arr[z_idx]   = Sv_m
            K_mD_arr[z_idx] = K_mD

    return Sv_arr, K_mD_arr


def calculate_ssa_volume(binary_vol, voxel_um):
    vox_m = voxel_um * 1e-6
    D, H, W = binary_vol.shape
    vol_m3  = D * H * W * vox_m**3
    eroded  = binary_erosion(binary_vol)
    surface = binary_vol.astype(int) - eroded.astype(int)
    sa_um2  = surface.sum() * vox_m**2 * 1e12
    vol_um3 = D * H * W * voxel_um**3
    ssa     = surface.sum() * vox_m**2 / vol_m3
    return {
        'surface_voxels'  : int(surface.sum()),
        'SSA_per_um'      : float(ssa),
        'SSA_m2_per_cm3'  : float(ssa * 1e4),
    }


def kozeny_carman_volume(phi_frac, Sv_m, tortuosity, shape_factor):
    if phi_frac <= 0 or phi_frac >= 1 or Sv_m <= 0:
        return {'K_m2': float('nan'), 'K_mD': float('nan')}
    num = phi_frac**3
    den = shape_factor * tortuosity**2 * Sv_m**2 * (1 - phi_frac)**2
    K_m2 = num / den
    return {'K_m2': float(K_m2), 'K_mD': float(K_m2 / 9.869233e-16)}


# ============================================================================
# CELL 10-11: VISUALISASI
# ============================================================================

def show_slice(ax, img2d, cmap='gray', vmin=None, vmax=None, norm=None,
               title='', xlabel='X (piksel)', ylabel='Y (piksel)', fontsize=8):
    img_disp = img2d.T
    kw = dict(cmap=cmap, aspect='auto', origin='lower')
    if norm is not None: kw['norm'] = norm
    else:
        if vmin is not None: kw['vmin'] = vmin
        if vmax is not None: kw['vmax'] = vmax
    im = ax.imshow(img_disp, **kw)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize-1); ax.set_ylabel(ylabel, fontsize=fontsize-1)
    return im


def plot_metric_vs_slice(ax, sl_idx, values_list, labels, colors,
                         title='', xlabel='', fill_between=False):
    for vals, lbl, clr in zip(values_list, labels, colors):
        ax.plot(vals, sl_idx, color=clr, lw=1, alpha=0.8, label=lbl)
    if fill_between and len(values_list) == 2:
        ax.fill_betweenx(sl_idx, values_list[0], values_list[1], alpha=0.15, color=colors[-1])
    ax.set_ylabel('Slice (Z)', fontsize=8); ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_ylim([sl_idx[0], sl_idx[-1]])


def visualize_results(test_noisy, test_denoise, test_clean,
                      bin_noisy, bin_denoise, bin_clean,
                      metrics, seg_noisy, seg_denoise,
                      por_clean, por_noisy, por_denoise,
                      ssa_clean, ssa_noisy, ssa_denoise,
                      kc_clean, kc_noisy, kc_denoise,
                      perm_c_sl, perm_n_sl, perm_d_sl,
                      psnr_list, ssim_list, rmse_list,
                      save_dir):

    n_test = len(test_clean); sl_idx = np.arange(n_test); mid = n_test // 2
    n_sl   = test_noisy[mid]; d_sl = test_denoise[mid]; c_sl = test_clean[mid]
    bn     = bin_noisy[mid];  bd   = bin_denoise[mid];  bc   = bin_clean[mid]
    err    = np.abs(c_sl - d_sl)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f'Perbandingan Slice Tengah (z={mid})', fontsize=12, fontweight='bold')
    show_slice(axes[0], n_sl, 'gray', 0, 1, title='Noisy (Input 0.8°)')
    show_slice(axes[1], d_sl, 'gray', 0, 1, title='Denoised (RED-CNN)')
    show_slice(axes[2], c_sl, 'gray', 0, 1, title='Clean GT (0.2°)')
    im = show_slice(axes[3], err, 'magma', 0, None, title=f'|Abs Error|\nMean={err.mean():.5f}')
    plt.colorbar(im, ax=axes[3], fraction=0.046); plt.tight_layout()
    p = os.path.join(save_dir, 'panel_A_image_compare.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle('Segmentasi Pori — Slice Tengah', fontsize=12, fontweight='bold')
    show_slice(axes[0], bn.astype(float), 'binary_r', title='Binary: Noisy')
    show_slice(axes[1], bd.astype(float), 'binary_r', title='Binary: Denoised')
    show_slice(axes[2], bc.astype(float), 'binary_r', title='Binary: Clean GT')
    overlay = np.zeros((*bd.shape, 3), float)
    overlay[..., 0] = bd.astype(float); overlay[..., 1] = bc.astype(float)
    axes[3].imshow(overlay.transpose(1, 0, 2), aspect='auto', origin='lower')
    axes[3].set_title('Overlay Pori\nKuning=TP  Merah=FP  Hijau=FN', fontsize=8)
    plt.tight_layout()
    p = os.path.join(save_dir, 'panel_B_segmentation.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Metrik Sinyal per Slice Test', fontsize=12, fontweight='bold')
    plot_metric_vs_slice(axes[0], sl_idx, [psnr_list], ['RED-CNN'], ['royalblue'],
                         title='PSNR per Slice (dB)', xlabel='PSNR (dB)')
    plot_metric_vs_slice(axes[1], sl_idx, [ssim_list], ['RED-CNN'], ['tomato'],
                         title='SSIM per Slice', xlabel='SSIM')
    plot_metric_vs_slice(axes[2], sl_idx, [rmse_list], ['RED-CNN'], ['purple'],
                         title='RMSE per Slice', xlabel='RMSE')
    plt.tight_layout()
    p = os.path.join(save_dir, 'panel_C_metrics_per_slice.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Petrofisika per Slice Test', fontsize=12, fontweight='bold')
    plot_metric_vs_slice(axes[0], sl_idx,
        [por_clean['porosity_per_slice']*100, por_noisy['porosity_per_slice']*100,
         por_denoise['porosity_per_slice']*100],
        ['Clean GT', 'Noisy', 'Denoised'], ['green', 'red', 'blue'],
        title='Porositas per Slice (%)', xlabel='Porosity (%)')
    plot_metric_vs_slice(axes[1], sl_idx,
        [perm_c_sl, perm_n_sl, perm_d_sl],
        ['Clean GT', 'Noisy', 'Denoised'], ['green', 'red', 'blue'],
        title='Permeabilitas per Slice (mD)\n[Kozeny-Carman]', xlabel='K (mD)')
    cats = ['Porosity\nClean', 'Porosity\nNoisy', 'Porosity\nDenoised']
    vals = [por_clean['porosity_percent'], por_noisy['porosity_percent'],
            por_denoise['porosity_percent']]
    axes[2].bar(cats, vals, color=['green', 'red', 'blue'], alpha=0.75)
    axes[2].set_title('Total Porosity (%)', fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(vals):
        axes[2].text(i, v + 0.05, f'{v:.2f}%', ha='center', fontsize=9)
    plt.tight_layout()
    p = os.path.join(save_dir, 'panel_D_petrophysics_per_slice.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")

    fig, ax = plt.subplots(figsize=(16, 7)); ax.axis('off')
    fig.suptitle('Ringkasan Hasil RED-CNN', fontsize=13, fontweight='bold')
    rows = [
        ['Metrik', 'Clean (GT)', 'Noisy (baseline)', 'Denoised (RED-CNN)', 'Δ (Den−Noisy)'],
        ['PSNR (dB)', '—', '—', f"{metrics['PSNR_mean']:.4f} ± {metrics['PSNR_std']:.4f}", '—'],
        ['SSIM', '—', '—', f"{metrics['SSIM_mean']:.4f} ± {metrics['SSIM_std']:.4f}", '—'],
        ['RMSE', '—', '—', f"{metrics['RMSE_mean']:.6f} ± {metrics['RMSE_std']:.6f}", '—'],
        ['Dice', '1.0000', f"{seg_noisy['Dice']:.4f}", f"{seg_denoise['Dice']:.4f}",
         f"{seg_denoise['Dice']-seg_noisy['Dice']:+.4f}"],
        ['IoU',  '1.0000', f"{seg_noisy['IoU']:.4f}",  f"{seg_denoise['IoU']:.4f}",
         f"{seg_denoise['IoU']-seg_noisy['IoU']:+.4f}"],
        ['Porosity (%)',
         f"{por_clean['porosity_percent']:.3f}", f"{por_noisy['porosity_percent']:.3f}",
         f"{por_denoise['porosity_percent']:.3f}",
         f"{por_denoise['porosity_percent']-por_noisy['porosity_percent']:+.3f}"],
        ['SSA (m²/cm³)',
         f"{ssa_clean['SSA_m2_per_cm3']:.4f}", f"{ssa_noisy['SSA_m2_per_cm3']:.4f}",
         f"{ssa_denoise['SSA_m2_per_cm3']:.4f}",
         f"{ssa_denoise['SSA_m2_per_cm3']-ssa_noisy['SSA_m2_per_cm3']:+.4f}"],
        ['K (mD) Kozeny-Carman',
         f"{kc_clean['K_mD']:.4e}", f"{kc_noisy['K_mD']:.4e}",
         f"{kc_denoise['K_mD']:.4e}",
         f"{kc_denoise['K_mD']-kc_noisy['K_mD']:+.4e}"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.9)
    for j in range(5):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    plt.tight_layout()
    p = os.path.join(save_dir, 'panel_E_summary_table.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(c_sl.ravel(), bins=64, alpha=0.5, color='green', density=True, label='Clean GT')
    ax.hist(n_sl.ravel(), bins=64, alpha=0.5, color='red',   density=True, label='Noisy')
    ax.hist(d_sl.ravel(), bins=64, alpha=0.5, color='blue',  density=True, label='Denoised')
    ax.set_title(f'Histogram Intensitas — Slice tengah (z={mid})', fontsize=10)
    ax.set_xlabel('Intensitas (ternormalisasi)'); ax.set_ylabel('Densitas')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, 'panel_F_histogram.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")


# ============================================================================
# FUNGSI UTAMA — BUNGKUSAN EKSEKUSI
# ============================================================================

def main():
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("  RED-CNN CONFIGURATION (OPTIMIZED)")
    print("="*65)
    print(f"  Training pair   : {cfg.TRAINING_PAIR}")
    print(f"  Patch size      : {cfg.PATCH_SIZE}×{cfg.PATCH_SIZE}")
    print(f"  Batch size      : {cfg.BATCH_SIZE}")
    print(f"  Epochs          : {cfg.EPOCHS}")
    print(f"  Loss            : {cfg.LOSS_TYPE}")
    print(f"  AMP             : {cfg.USE_AMP}")
    print(f"  torch.compile   : {cfg.USE_COMPILE}")
    print(f"  Workers         : {cfg.NUM_WORKERS} (CPU cores: {CPU_COUNT})")
    print(f"  Tracker interval: {cfg.TRACKER_INTERVAL}s")
    print(f"  Save dir        : {cfg.SAVE_DIR}")

    # Uji arsitektur di sini agar tidak alokasi VRAM secara global
    _t   = torch.randn(2, 1, 128, 128).to(device)
    _m   = REDCNN().to(device)
    _out = _m(_t)
    assert _out.shape == _t.shape, "Shape mismatch!"
    print(f"\n✓ RED-CNN architecture OK — params: {count_parameters(_m):,}")
    del _t, _out, _m
    torch.cuda.empty_cache()

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  LOADING DATA")
    print("="*65)

    vol_clean = load_and_crop_volume(cfg.PATH_02, cfg.CROP_SIZE, cfg.CROP_ORIGIN)
    vol_08    = load_and_crop_volume(cfg.PATH_08, cfg.CROP_SIZE, cfg.CROP_ORIGIN)

    if cfg.TRAINING_PAIR in ('0.4->0.2', 'both'):
        vol_04 = load_and_crop_volume(cfg.PATH_04, cfg.CROP_SIZE, cfg.CROP_ORIGIN)

    vol_clean_n, c_p1, c_p99 = normalize_volume(vol_clean)
    vol_08_n,    n_p1, n_p99  = normalize_volume(vol_08)

    norm_stats = {
        'p1_02': float(c_p1), 'p99_02': float(c_p99),
        'p1_08': float(n_p1), 'p99_08': float(n_p99),
    }
    np.save(os.path.join(cfg.SAVE_DIR, 'norm_stats.npy'), norm_stats)

    n_slices = vol_clean_n.shape[0]
    train_idx, val_idx, test_idx = split_slices(n_slices, cfg.TRAIN_FRAC, cfg.VAL_FRAC)
    print(f"\n  Total slices  : {n_slices}")
    print(f"  Train slices  : {len(train_idx)}")
    print(f"  Val slices    : {len(val_idx)}")
    print(f"  Test slices   : {len(test_idx)}")

    train_ds = PatchDataset(vol_08_n, vol_clean_n, train_idx,
                            cfg.PATCH_SIZE, cfg.PATCHES_PER_SLICE, cfg.AUGMENT, 'train')
    val_ds   = PatchDataset(vol_08_n, vol_clean_n, val_idx,
                            cfg.PATCH_SIZE, cfg.PATCHES_PER_SLICE, False, 'val')

    _loader_kwargs = dict(
        batch_size         = cfg.BATCH_SIZE,
        num_workers        = cfg.NUM_WORKERS,
        pin_memory         = True,
        persistent_workers = True,
        prefetch_factor    = 2,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True, **_loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False,                 **_loader_kwargs)

    print(f"\n  DataLoader workers : {cfg.NUM_WORKERS}")
    print(f"  Train patches : {len(train_ds):,}  ({len(train_loader):,} batches)")
    print(f"  Val patches   : {len(val_ds):,}  ({len(val_loader):,} batches)")

    # ── RESOURCE TRACKER ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  INISIALISASI RESOURCE TRACKER")
    print("="*65)

    tracker = ResourceTracker(
        interval = cfg.TRACKER_INTERVAL,
        log_path = os.path.join(cfg.SAVE_DIR, 'resource_log.csv')
    )
    tracker.start()

    # ── TRAINING ──────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  TRAINING RED-CNN")
    print("="*65)
    model, history, model_path = train_model(cfg, tracker, train_loader, val_loader, norm_stats)
    plot_training_history(history, cfg.SAVE_DIR)

    # ── INFERENCE ─────────────────────────────────────────────────────────────
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"\n  Loaded best model: epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")

    denoised_test, metrics, psnr_list, ssim_list, rmse_list = \
        run_inference_testset(model, vol_08_n, vol_clean_n, test_idx, cfg, device, tracker)

    # ── PETROPHYSICS ──────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PETROPHYSICAL ANALYSIS (PARALEL — ProcessPoolExecutor)")
    print("="*65)

    test_clean   = vol_clean_n[test_idx]
    test_noisy   = vol_08_n[test_idx]
    test_denoise = denoised_test

    bin_clean,   thresh_c = binarize_volume(test_clean)
    bin_noisy,   thresh_n = binarize_volume(test_noisy)
    bin_denoise, thresh_d = binarize_volume(test_denoise)

    por_clean   = calculate_porosity(bin_clean)
    por_noisy   = calculate_porosity(bin_noisy)
    por_denoise = calculate_porosity(bin_denoise)

    seg_noisy   = calculate_dice_iou(bin_noisy,   bin_clean)
    seg_denoise = calculate_dice_iou(bin_denoise, bin_clean)

    ssa_clean   = calculate_ssa_volume(bin_clean,   cfg.VOXEL_SIZE_UM)
    ssa_noisy   = calculate_ssa_volume(bin_noisy,   cfg.VOXEL_SIZE_UM)
    ssa_denoise = calculate_ssa_volume(bin_denoise, cfg.VOXEL_SIZE_UM)

    vox_m   = cfg.VOXEL_SIZE_UM * 1e-6
    D, H, W = bin_clean.shape
    vol_m3  = D * H * W * vox_m**3

    def _vol_Sv(bvol):
        surf = bvol.astype(int) - binary_erosion(bvol).astype(int)
        return surf.sum() * vox_m**2 / vol_m3

    kc_clean   = kozeny_carman_volume(por_clean['porosity_total'],   _vol_Sv(bin_clean),
                                      cfg.KC_TORTUOSITY, cfg.KC_SHAPE_FACTOR)
    kc_noisy   = kozeny_carman_volume(por_noisy['porosity_total'],   _vol_Sv(bin_noisy),
                                      cfg.KC_TORTUOSITY, cfg.KC_SHAPE_FACTOR)
    kc_denoise = kozeny_carman_volume(por_denoise['porosity_total'], _vol_Sv(bin_denoise),
                                      cfg.KC_TORTUOSITY, cfg.KC_SHAPE_FACTOR)

    with tracker.phase("petrophysics"):
        _, perm_c_sl = run_petrophysics_parallel(
            bin_clean,   por_clean,   cfg.VOXEL_SIZE_UM,
            cfg.KC_TORTUOSITY, cfg.KC_SHAPE_FACTOR, "Clean", tracker)
        _, perm_n_sl = run_petrophysics_parallel(
            bin_noisy,   por_noisy,   cfg.VOXEL_SIZE_UM,
            cfg.KC_TORTUOSITY, cfg.KC_SHAPE_FACTOR, "Noisy", tracker)
        _, perm_d_sl = run_petrophysics_parallel(
            bin_denoise, por_denoise, cfg.VOXEL_SIZE_UM,
            cfg.KC_TORTUOSITY, cfg.KC_SHAPE_FACTOR, "Denoised", tracker)

    # ── VISUALISASI ───────────────────────────────────────────────────────────
    visualize_results(
        test_noisy=test_noisy, test_denoise=test_denoise, test_clean=test_clean,
        bin_noisy=bin_noisy, bin_denoise=bin_denoise, bin_clean=bin_clean,
        metrics=metrics, seg_noisy=seg_noisy, seg_denoise=seg_denoise,
        por_clean=por_clean, por_noisy=por_noisy, por_denoise=por_denoise,
        ssa_clean=ssa_clean, ssa_noisy=ssa_noisy, ssa_denoise=ssa_denoise,
        kc_clean=kc_clean, kc_noisy=kc_noisy, kc_denoise=kc_denoise,
        perm_c_sl=perm_c_sl, perm_n_sl=perm_n_sl, perm_d_sl=perm_d_sl,
        psnr_list=psnr_list, ssim_list=ssim_list, rmse_list=rmse_list,
        save_dir=cfg.SAVE_DIR,
    )

    # ── SIMPAN OUTPUT ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  MENYIMPAN OUTPUT KE DISK LOKAL")
    print("="*65)

    out_tif = os.path.join(cfg.SAVE_DIR, 'denoised_test_slices.tif')
    imwrite(out_tif, denoised_test)
    print(f"  ✓ Denoised TIFF (float32) : {out_tif}  ({os.path.getsize(out_tif)/1e6:.1f} MB)")

    out_8 = os.path.join(cfg.SAVE_DIR, 'denoised_test_slices_8bit.tif')
    imwrite(out_8, (denoised_test * 255).astype(np.uint8))
    print(f"  ✓ Denoised TIFF (uint8)   : {out_8}")

    df = pd.DataFrame({
        'slice_z'          : test_idx,
        'PSNR_dB'          : psnr_list,
        'SSIM'             : ssim_list,
        'RMSE'             : rmse_list,
        'Porosity_clean_%' : por_clean['porosity_per_slice']   * 100,
        'Porosity_noisy_%' : por_noisy['porosity_per_slice']   * 100,
        'Porosity_den_%'   : por_denoise['porosity_per_slice'] * 100,
        'Perm_clean_mD'    : perm_c_sl,
        'Perm_noisy_mD'    : perm_n_sl,
        'Perm_den_mD'      : perm_d_sl,
    })
    csv_path = os.path.join(cfg.SAVE_DIR, 'metrics_per_slice.csv')
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Metrics CSV             : {csv_path}")

    summary = {
        'model'       : 'RED-CNN',
        'training_pair': cfg.TRAINING_PAIR,
        'timestamp'   : datetime.now().isoformat(),
        'optimizations': {
            'AMP'          : cfg.USE_AMP,
            'torch_compile': cfg.USE_COMPILE,
            'num_workers'  : cfg.NUM_WORKERS,
            'cudnn_benchmark': True,
        },
        'test_metrics': metrics,
        'porosity_%': {
            'clean': por_clean['porosity_percent'],
            'noisy': por_noisy['porosity_percent'],
            'denoised': por_denoise['porosity_percent'],
        },
        'SSA_m2_per_cm3': {
            'clean': ssa_clean['SSA_m2_per_cm3'],
            'noisy': ssa_noisy['SSA_m2_per_cm3'],
            'denoised': ssa_denoise['SSA_m2_per_cm3'],
        },
        'Permeability_mD_KozenyCarman': {
            'clean': kc_clean['K_mD'], 'noisy': kc_noisy['K_mD'],
            'denoised': kc_denoise['K_mD'],
            'delta_den_minus_noisy': kc_denoise['K_mD'] - kc_noisy['K_mD'],
        },
        'pore_segmentation': {
            'Dice_noisy': seg_noisy['Dice'],   'IoU_noisy': seg_noisy['IoU'],
            'Dice_denoised': seg_denoise['Dice'], 'IoU_denoised': seg_denoise['IoU'],
        },
    }
    json_path = os.path.join(cfg.SAVE_DIR, 'results_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary JSON            : {json_path}")

    tracker.stop()
    tracker.summary()
    tracker.plot(save_path=os.path.join(cfg.SAVE_DIR, 'resource_usage.png'))

    print("\n" + "="*65)
    print(f"  SELESAI — semua output tersimpan di: {cfg.SAVE_DIR}")
    print("="*65)


# ============================================================================
# ENTRY POINT — WAJIB untuk ProcessPoolExecutor & PyTorch DataLoader di Windows
# ============================================================================

if __name__ == '__main__':
    # Opsional namun sangat direkomendasikan untuk Windows OS
    multiprocessing.freeze_support()
    
    # Menjalankan keseluruhan logika
    main()