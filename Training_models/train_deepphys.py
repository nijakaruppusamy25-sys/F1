"""
train_deepphys.py
=================
Training pipeline for the DeepPhys rPPG model.

DATASET EXPECTED:
    Same structure as your existing project:
        data/X.npy   — shape (N, 150)   raw green-channel signal per video
        data/y.npy   — shape (N,)       BPM label per video

    Because DeepPhys operates on frame *pairs* (appearance + motion),
    this script reconstructs synthetic frame pairs from the 1-D signal
    stored in X.npy.  Each consecutive pair (t, t-1) becomes one sample.

    For best results, collect RGB frame pairs directly (see NOTE below).

NOTE — upgrading to full RGB frame pairs:
    Swap the SyntheticPPGDataset below with RGBFrameDataset (stub provided)
    once you have actual ROI crops saved as NumPy arrays or video frames.
    The model, training loop, and inference script need zero changes.

OUTPUT:
    models/deepphys_model.pth   — saved state dict
    models/deepphys_loss.png    — training / validation loss curve (optional)

USAGE:
    python train_deepphys.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from deepphys_model import DeepPhys, preprocess_frame_pair


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE      = 36       # spatial size fed to DeepPhys
BATCH_SIZE    = 8
EPOCHS        = 200
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.15     # fraction of data held out for validation
SEED          = 42

os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/deepphys_model.pth"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET  —  synthetic from 1-D green-channel signal
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticPPGDataset(Dataset):
    """
    Builds DeepPhys-compatible (appearance, motion, bvp) triplets from
    the 1-D green-channel signals already stored in data/X.npy.

    Each video signal of length T yields (T-1) frame-pair samples.
    The scalar BVP target is the signal difference  x[t] - x[t-1],
    normalised to zero mean / unit std within the video.

    This is an *approximation* — a proper dataset would store actual
    RGB ROI crops.  The model architecture is identical either way.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, img_size: int = 36):
        """
        Parameters
        ----------
        X        : (N, T)  float  green-channel signal per video
        y        : (N,)    float  BPM label per video
        img_size : int     spatial size for synthetic "frames"
        """
        self.img_size = img_size
        self.samples  = []   # list of (appearance, motion, bvp_target)

        rng = np.random.default_rng(SEED)

        for signal, bpm in zip(X, y):

            T = len(signal)

            # Normalise signal within video
            sig_norm = (signal - signal.mean()) / (signal.std() + 1e-8)

            for t in range(1, T):

                bvp_target = float(sig_norm[t] - sig_norm[t - 1])

                # ── Synthesise appearance and motion "frames" ──────────────
                # We tile the scalar intensity into a (img_size, img_size, 3)
                # patch, then add spatial noise to give the CNN something to
                # learn.  This is only needed when real ROI crops are absent.
                v_t   = signal[t]   / 255.0
                v_tm1 = signal[t-1] / 255.0

                base_t   = np.full((img_size, img_size, 3), v_t,   dtype=np.float32)
                base_tm1 = np.full((img_size, img_size, 3), v_tm1, dtype=np.float32)

                noise = rng.normal(0, 0.02, base_t.shape).astype(np.float32)
                base_t   = np.clip(base_t   + noise, 0, 1)
                base_tm1 = np.clip(base_tm1 + noise * 0.9, 0, 1)

                appearance = base_t.transpose(2, 0, 1)            # (3, H, W)

                motion = (base_t - base_tm1) / (base_tm1 + 1e-3)
                motion = np.clip(motion, -1.0, 1.0).transpose(2, 0, 1)

                self.samples.append((appearance, motion, np.float32(bvp_target)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        appearance, motion, bvp = self.samples[idx]
        return (
            torch.from_numpy(appearance),
            torch.from_numpy(motion),
            torch.tensor(bvp),
        )


# ─────────────────────────────────────────────────────────────────────────────
# DATASET STUB — real RGB frame pairs  (swap in when ready)
# ─────────────────────────────────────────────────────────────────────────────

class RGBFrameDataset(Dataset):
    """
    Use this dataset once you have actual ROI crops saved.

    Expected files:
        data/frames_t.npy    — (N, H, W, 3)  uint8  current frame ROI
        data/frames_tm1.npy  — (N, H, W, 3)  uint8  previous frame ROI
        data/bvp_targets.npy — (N,)           float  BVP amplitude label

    To generate bvp_targets, bandpass-filter the green signal and take
    consecutive differences:
        from scipy.signal import butter, filtfilt
        bvp = filtfilt(*butter(3,[0.7,4.0],btype='band',fs=30), green_signal)
        bvp_targets = np.diff(bvp)
    """

    def __init__(self, img_size: int = 36):
        self.img_size = img_size
        self.frames_t   = np.load("data/frames_t.npy")
        self.frames_tm1 = np.load("data/frames_tm1.npy")
        self.targets    = np.load("data/bvp_targets.npy").astype(np.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        app, mot = preprocess_frame_pair(
            self.frames_t[idx],
            self.frames_tm1[idx],
            self.img_size,
        )
        return (
            torch.from_numpy(app),
            torch.from_numpy(mot),
            torch.tensor(self.targets[idx]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> float:

    model.train()
    total_loss = 0.0

    for appearance, motion, bvp_target in loader:

        appearance  = appearance.to(device)
        motion      = motion.to(device)
        bvp_target  = bvp_target.to(device).unsqueeze(1)   # (B,) → (B, 1)

        optimizer.zero_grad()

        pred = model(appearance, motion)          # (B, 1)
        loss = criterion(pred, bvp_target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> float:

    model.eval()
    total_loss = 0.0

    for appearance, motion, bvp_target in loader:

        appearance  = appearance.to(device)
        motion      = motion.to(device)
        bvp_target  = bvp_target.to(device).unsqueeze(1)

        pred = model(appearance, motion)
        loss = criterion(pred, bvp_target)
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Device ───────────────────────────────────────────────────────────────
    # MPS (Apple Silicon) → CPU fallback for full macOS compatibility
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU) ✅")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # ── Load signals ─────────────────────────────────────────────────────────
    print("\nLoading dataset...")

    X = np.load("data/X.npy")
    y = np.load("data/y.npy")

    print(f"  Signals loaded — X: {X.shape}  y: {y.shape}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    dataset = SyntheticPPGDataset(X, y, img_size=IMG_SIZE)
    print(f"  Frame-pair samples: {len(dataset):,}")

    # ── Train / val split ─────────────────────────────────────────────────────
    n_val   = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {n_train:,}  |  Val: {n_val:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DeepPhys(img_size=IMG_SIZE).to(device)
    print(f"\nDeepPhys — trainable parameters: {model.count_parameters():,}")

    # ── Loss & Optimiser ──────────────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs...")
    print("-" * 52)

    best_val_loss = float("inf")
    train_losses  = []
    val_losses    = []

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            ckpt_tag = " ← best"
        else:
            ckpt_tag = ""

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:>4}/{EPOCHS}]  "
                f"Train Loss: {train_loss:.6f}  "
                f"Val Loss: {val_loss:.6f}"
                f"{ckpt_tag}"
            )

    print("-" * 52)
    print(f"\nTraining Complete ✅")
    print(f"Best val loss : {best_val_loss:.6f}")
    print(f"Model saved   → {MODEL_PATH}")

    # ── Optional: save loss curve ─────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="Train loss", linewidth=1.2)
        ax.plot(val_losses,   label="Val loss",   linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("DeepPhys Training Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        curve_path = "models/deepphys_loss.png"
        fig.savefig(curve_path, dpi=120)
        print(f"Loss curve saved → {curve_path}")
        plt.close(fig)

    except ImportError:
        print("(matplotlib not found — skipping loss curve plot)")


if __name__ == "__main__":
    main()
