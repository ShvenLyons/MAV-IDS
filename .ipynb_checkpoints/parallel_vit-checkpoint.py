import os, time, math, random
import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -------------------- configuration --------------------
SEED       = 42
MAX_LEN    = 1500
CSV_PATH   = "data/Payload_data_CICIDS2017_train_split_binary.csv"
SAVE_DIR   = "model/checkpoints/byte_vit_cicids"
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS     = 5
BATCH_SIZE = 256
LR         = 1e-3

MODEL_TYPE   = "pvit"   # "vit" / "pvit"
PVIT_BRANCH  = 2

IMG_N      = math.ceil(math.sqrt(MAX_LEN))  # 39
PATCH_SIZE = 3                              
EMB_DIM    = 192
DEPTH      = 6
NUM_HEADS  = 3
MLP_RATIO  = 4.0
DROPOUT    = 0.1

# -------------------- seed --------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)

# -------------------- data_loading --------------------
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    payload_cols = [f"payload_byte_{i}" for i in range(1, MAX_LEN + 1)]
    assert all(c in df.columns for c in payload_cols), "CSV 缺少 payload_byte_* 列"
    assert "label" in df.columns, "CSV 缺少 label 列"
    X = df[payload_cols].fillna(0).to_numpy(dtype=np.int64)
    X = np.clip(X, 0, 255)  # 0-255
    y = df["label"].astype(int).to_numpy()

    unique, counts = np.unique(y, return_counts=True)
    min_n = counts.min()
    idxs_balanced = []
    for cls in unique:
        cls_idx = np.where(y == cls)[0]
        sel = np.random.choice(cls_idx, size=min_n, replace=False)
        idxs_balanced.append(sel)
    idxs_balanced = np.concatenate(idxs_balanced)
    np.random.shuffle(idxs_balanced)
    X_bal, y_bal = X[idxs_balanced], y[idxs_balanced]

    # 70/20/10（train/val/test）
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_bal, y_bal, test_size=0.30, random_state=SEED, stratify=y_bal
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=2/3, random_state=SEED, stratify=y_tmp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# -------------------- 边缘螺旋 → RGB --------------------
def spiral_indices(n: int) -> np.ndarray:
    """返回按“外圈→内圈、顺时针”的扁平索引序列（长度 n*n）。"""
    mat = np.zeros((n, n), dtype=np.int32)
    top, left, bottom, right = 0, 0, n - 1, n - 1
    k = 0
    while left <= right and top <= bottom:
        for j in range(left, right + 1):    # →
            mat[top, j] = k; k += 1
        for i in range(top + 1, bottom + 1):  # ↓
            mat[i, right] = k; k += 1
        if top < bottom:
            for j in range(right - 1, left - 1, -1):  # ←
                mat[bottom, j] = k; k += 1
        if left < right:
            for i in range(bottom - 1, top, -1):      # ↑
                mat[i, left] = k; k += 1
        top += 1; left += 1; bottom -= 1; right -= 1
    flat_idx = np.argsort(mat.reshape(-1))
    return flat_idx

SPIRAL_IDX = spiral_indices(IMG_N)

class ByteRGBDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.uint8)
        self.y = y.astype(np.int64)
        self.n = IMG_N
        self.flat_len = self.n * self.n

    def __len__(self): return len(self.X)

    def _bytes_to_rgb(self, arr1d: np.ndarray) -> np.ndarray:
        buf = np.zeros(self.flat_len, dtype=np.uint8)
        L = min(len(arr1d), self.flat_len)
        buf[:L] = arr1d[:L]

        # 按螺旋序写入矩阵
        mat_flat = np.zeros_like(buf)
        mat_flat[SPIRAL_IDX] = buf
        mat = mat_flat.reshape(self.n, self.n)

        # 构造 RGB：原图、90°、180°
        g = np.rot90(mat, k=1)
        b = np.rot90(mat, k=2)
        rgb = np.stack([mat, g, b], axis=0)   # (3, H, W)
        return rgb.astype(np.float32) / 255.0

    def __getitem__(self, idx):
        x = self._bytes_to_rgb(self.X[idx])
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y)

# -------------------- ViT 实现（Post-Norm + Mean Pooling） --------------------
class PatchEmbed(nn.Module):
    """
    不使用 CLS；仅位置编码。输出 (B, Np, D)
    """
    def __init__(self, img_size=IMG_N, patch=PATCH_SIZE, in_chans=3, embed_dim=EMB_DIM):
        super().__init__()
        assert img_size % patch == 0, "img_size 必须能被 patch_size 整除"
        self.grid = img_size // patch
        self.num_patches = self.grid * self.grid
        self.patch = patch
        self.proj = nn.Linear(in_chans * patch * patch, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):  # x: (B,3,H,W)
        B, C, H, W = x.shape
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)          # (B, C, g, g, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()   # (B, g, g, C, p, p)
        x = x.view(B, self.grid * self.grid, -1)       # (B, Np, C*p*p)
        x = self.proj(x)                                # (B, Np, D)
        x = x + self.pos_embed                          # (B, Np, D)
        return x

class ViTBackbone(nn.Module):
    """
    ViT 编码骨干（Post-Norm），读出为 mean pooling（不含 CLS）
    """
    def __init__(self, embed_dim=EMB_DIM, num_heads=NUM_HEADS, depth=DEPTH,
                 mlp_ratio=MLP_RATIO, dropout=DROPOUT):
        super().__init__()
        self.stem = PatchEmbed(embed_dim=embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=False  # Post-Norm 对齐图示
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):                 # x: (B,3,H,W)
        tok = self.stem(x)                # (B, Np, D)
        h = self.encoder(tok)             # (B, Np, D)
        rep = h.mean(dim=1)               # Mean Pooling 作为读出 (B, D)
        return rep

class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = ViTBackbone()
        self.head = nn.Linear(EMB_DIM, num_classes)

    def forward(self, x):
        rep = self.backbone(x)
        return self.head(rep)

class ParallelViTClassifier(nn.Module):
    """
    并行 ViT：每个分支拥有独立的 Patch/Pos + Encoder；输出拼接后接 Linear。
    """
    def __init__(self, num_classes: int, branches: int = 2):
        super().__init__()
        self.backbones = nn.ModuleList([ViTBackbone() for _ in range(branches)])
        self.head = nn.Linear(EMB_DIM * branches, num_classes)

    def forward(self, x):
        reps = [bb(x) for bb in self.backbones]  # list of (B, D)
        rep = torch.cat(reps, dim=-1)            # (B, D*B)
        return self.head(rep)

# -------------------- 训练 & 评估 --------------------
@dataclass
class Metrics:
    loss: float
    acc: float

def accuracy(pred, target):
    return (pred.argmax(dim=1) == target).float().mean().item()

def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        bs = y.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy(logits.detach(), y) * bs
        n += bs
    return Metrics(running_loss / n, running_acc / n)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = y.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy(logits, y) * bs
        n += bs
    return Metrics(running_loss / n, running_acc / n)

@torch.no_grad()
def inference_timing(model, loader, device):
    model.eval()

    # End-to-end（含 dataloader+拷贝+前向）
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_items = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        n_items += x.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    end_to_end = (t1 - t0) / n_items * 1000.0

    # 仅模型前向（CUDA events）
    n_items2 = 0
    total_ms = 0.0
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            total_ms += starter.elapsed_time(ender)
        else:
            s = time.perf_counter()
            _ = model(x)
            total_ms += (time.perf_counter() - s) * 1000.0
        n_items2 += x.size(0)
    model_only = total_ms / n_items2
    return end_to_end, model_only

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) 数据
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_and_prepare(CSV_PATH)
    num_classes = int(np.max(y_tr)) + 1
    ds_tr = ByteRGBDataset(X_tr, y_tr)
    ds_va = ByteRGBDataset(X_va, y_va)
    ds_te = ByteRGBDataset(X_te, y_te)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=(device.type=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=0, pin_memory=(device.type=="cuda"))
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=0, pin_memory=(device.type=="cuda"))

    # 2) 模型
    if MODEL_TYPE == "vit":
        model = ViTClassifier(num_classes)
        ckpt_name = "vit_postnorm_mean_best.pt"
    else:
        model = ParallelViTClassifier(num_classes, branches=PVIT_BRANCH)
        ckpt_name = f"pvit{PVIT_BRANCH}_postnorm_mean_best.pt"
    model.to(device)

    # 3) 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1.0
    for ep in range(1, EPOCHS + 1):
        tr = train_one_epoch(model, dl_tr, optimizer, criterion, device)
        va = evaluate(model, dl_va, criterion, device)
        print(f"[Epoch {ep}/{EPOCHS}] "
              f"train_loss={tr.loss:.4f} acc={tr.acc:.4f} | "
              f"val_loss={va.loss:.4f} acc={va.acc:.4f}")
        if va.acc > best_val:
            best_val = va.acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, ckpt_name))

    # 4) 评估
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, ckpt_name), map_location=device))
    te = evaluate(model, dl_te, criterion, device)
    print(f"[Test] loss={te.loss:.4f} acc={te.acc:.4f}")

    # 5) 时间
    end2end_ms, model_ms = inference_timing(model, dl_te, device)
    print(f"[Inference] End-to-end: {end2end_ms:.3f} ms/item | Model-only: {model_ms:.3f} ms/item")
    print(f"Throughput (model-only): {1000.0 / model_ms:.2f} items/s")

if __name__ == "__main__":
    main()
