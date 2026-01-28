import os, time, math, random
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ========== Config ==========
SEED = 42
MAX_LEN = 1500
CSV_PATH = "data/CICIDS/Payload_data_CICIDS2017_train_split_binary.csv"
SAVE_DIR = "model/checkpoints/ml_baselines"
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LABELS = 2

os.makedirs(SAVE_DIR, exist_ok=True)

# ========== Args ==========
def parse_args():
    parser = argparse.ArgumentParser(description="ML baselines (DNN/CNN/RNN/LSTM) for byte payloads")
    parser.add_argument(
        "model_type",
        nargs="?",
        default="cnn",
        choices=["cnn", "dnn", "rnn", "lstm", "cnn1d", "cnn2d"],
        help="Model type to use (default: cnn)"
    )

    parser.add_argument(
        "--model_type",
        dest="model_type_kw",
        choices=["cnn", "dnn", "rnn", "lstm", "cnn1d", "cnn2d"],
        help="Model type to use (overrides positional if provided)"
    )
    args = parser.parse_args()
    model_type = args.model_type_kw if args.model_type_kw is not None else args.model_type
    return model_type

# ========== Seed ==========
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)

# ========== Dataset ==========
class ByteDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========== Data load & preprocess ==========
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    payload_cols = [f"payload_byte_{i}" for i in range(1, MAX_LEN + 1)]
    assert all(c in df.columns for c in payload_cols), "缺少 payload_byte_* 列"
    assert "label" in df.columns, "缺少 label 列"

    X = df[payload_cols].fillna(0).to_numpy(dtype=np.float32)
    X = np.clip(X, 0, 255) / 255.0
    y = df["label"].astype(int).to_numpy()

    # 均衡采样
    unique, counts = np.unique(y, return_counts=True)
    min_n = counts.min()
    idxs_bal = np.concatenate([
        np.random.choice(np.where(y == cls)[0], size=min_n, replace=False)
        for cls in unique
    ])
    np.random.shuffle(idxs_bal)
    X_bal, y_bal = X[idxs_bal], y[idxs_bal]

    # 70/20/10 划分
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_bal, y_bal, test_size=0.30, random_state=SEED, stratify=y_bal)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=2/3, random_state=SEED, stratify=y_tmp)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ========== Models ==========
# 1. DNN
class DNN(nn.Module):
    def __init__(self, input_dim=MAX_LEN, hidden=HIDDEN_SIZE, num_labels=NUM_LABELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_labels)
        )
    def forward(self, x): 
        return self.net(x)

# 2. CNN
class CNN(nn.Module):
    # num_convs卷积块数量; base_channels 第一层卷积输出通道; channel_mult 通道数倍率
    def __init__(self, input_len=MAX_LEN, num_labels=NUM_LABELS, num_convs: int = 2, base_channels: int = 32, channel_mult: int = 2, kernel_size: int = 5):
        super().__init__()
        conv_layers = []
        in_channels = 1
        out_channels = base_channels
        current_len = input_len
        for i in range(num_convs):
            conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels = out_channels * channel_mult
            current_len = current_len // 2
        self.conv = nn.Sequential(*conv_layers)
        last_channels = in_channels
        fc_in_dim = current_len * last_channels
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels),
        )
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.conv(x)    # (B, C, L')
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 3. RNN
class RNN(nn.Module):
    def __init__(self, input_size=MAX_LEN, hidden=HIDDEN_SIZE, num_labels=NUM_LABELS):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden, num_labels)
    def forward(self, x):
        x = x.unsqueeze(1)
        _, h = self.rnn(x)
        return self.fc(h[-1])

# 4. LSTM
class LSTM(nn.Module):
    def __init__(self, hidden=HIDDEN_SIZE, num_labels=NUM_LABELS, seq_len: int | None = None, input_size: int = 1, num_layers: int = 2,):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden,
            batch_first=True,
            num_layers=self.num_layers,
        )
        self.fc = nn.Linear(self.hidden, num_labels)
    def forward(self, x):
        B, L = x.shape
        if self.seq_len is None:
            x = x.unsqueeze(-1)  # (B, L, 1)
        else:
            assert (
                self.seq_len * self.input_size == L
            ), f"seq_len * input_size = {self.seq_len} * {self.input_size} != input length {L}"
            x = x.view(B, self.seq_len, self.input_size)  # (B, seq_len, input_size)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

# 5. 1D-CNN
class CNN_1D(nn.Module):
    def __init__(self, input_len=MAX_LEN, num_labels=NUM_LABELS):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.global_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(128, num_labels)
    def forward(self, x):
        x = x.unsqueeze(1)           # (B, 1, L)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)            # (B, 64, L/2)
        x = torch.relu(self.conv2(x))# (B, 128, L/2)
        x = self.global_pool(x)      # (B, 128, 1)
        x = x.squeeze(-1)            # (B, 128)
        x = self.fc(x)               # (B, num_labels)
        return x

# 6. 2D-CNN
class CNN_2D(nn.Module):
    def __init__(self, num_labels=NUM_LABELS, h=30, w=50):
        super().__init__()
        self.h = h
        self.w = w
        assert self.h * self.w == MAX_LEN, \
            f"2D-CNN needs h*w == MAX_LEN, Current {self.h}*{self.w} != {MAX_LEN}"
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.global_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.fc = nn.Linear(32, num_labels)
    def forward(self, x):
        B, L = x.shape
        assert L == self.h * self.w, f"Input-Lenth {L} is different from {self.h}*{self.w}."
        # reshape
        x = x.view(B, 1, self.h, self.w)    # (B,1,30,50)
        x = torch.relu(self.conv1(x))       # (B,16,30,50)
        x = self.pool1(x)                   # (B,16,15,25)
        x = torch.relu(self.conv2(x))       # (B,32,15,25)
        x = self.global_pool(x)             # (B,32,1,1)
        x = x.view(B, 32)                   # (B,32)
        x = self.fc(x)                      # (B,num_labels)
        return x

def build_model(model_type):
    if model_type == "cnn":
        return CNN(input_len=MAX_LEN, num_labels=NUM_LABELS, num_convs=2, base_channels=32, channel_mult=2)
    elif model_type == "rnn":
        return RNN()
    elif model_type == "lstm":
        return LSTM(hidden=HIDDEN_SIZE, num_labels=NUM_LABELS, seq_len=1500, input_size=1)
    elif model_type == "cnn1d":
        return CNN_1D(input_len=MAX_LEN, num_labels=NUM_LABELS)
    elif model_type == "cnn2d":
        return CNN_2D(num_labels=NUM_LABELS, h=30, w=50)
    else:
        return DNN()

# ========== Train / Eval ==========
def accuracy(pred, target):
    return (pred.argmax(dim=1) == target).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    report = classification_report(y_true, y_pred, digits=4)
    return total_loss / n, total_acc / n, report

@torch.no_grad()
def benchmark_inference(model, loader, device):
    model.eval()
    if device.type == "cuda": 
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for X, _ in loader:
        X = X.to(device, non_blocking=True)
        _ = model(X)
        n += X.size(0)
    if device.type == "cuda": 
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms_per_item = (t1 - t0) * 1000.0 / n
    return ms_per_item

# ========== Main ==========
def main():
    model_type = parse_args()
    print(f"Using model type: {model_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_and_prepare(CSV_PATH)
    print(f"train:{len(X_tr)} val:{len(X_va)} test:{len(X_te)}")

    train_loader = DataLoader(ByteDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ByteDataset(X_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(ByteDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1
    best_path = os.path.join(SAVE_DIR, f"{model_type}_best.pt")

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_acc, n = 0, 0, 0
        for X, y in tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, y) * bs
            n += bs

        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {ep}] train_loss={total_loss/n:.4f} acc={total_acc/n:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved best model to {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, report = evaluate(model, test_loader, criterion, device)
    print("\n=== Test ===")
    print(f"loss={test_loss:.4f} acc={test_acc:.4f}")
    print(report)

    ms_per_item = benchmark_inference(model, test_loader, device)
    # print(f"\n=== Inference Benchmark ===")
    # print(f"{model_type.upper()} | {ms_per_item:.3f} ms/item | Throughput: {1000.0/ms_per_item:.2f} items/s")

if __name__ == "__main__":
    main()