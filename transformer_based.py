import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random, math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
import time

# ========= configuration =========
SEED = 42
VOCAB_SIZE = 256
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
INTERMEDIATE_SIZE = 3072
MAX_LEN = 1500
NUM_LABELS = 2
LR = 2e-5
EPOCHS = 5
BATCH_SIZE = 128
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

CSV_PATH = "data/Payload_data_CICIDS2017_train_split_binary.csv"
SAVE_DIR = "model/checkpoints/byte_bert_cicids"

# ========= Seed =========
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(SEED)

# ========= dataset =========
class ByteDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.input_ids = torch.tensor(X, dtype=torch.long)
        self.attention_mask = torch.ones_like(self.input_ids, dtype=torch.long)
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

# ========= data_load & preprocess =========
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    # 只取 payload_byte_ 和 label
    payload_cols = [f"payload_byte_{i}" for i in range(1, MAX_LEN + 1)]
    assert all(c in df.columns for c in payload_cols), "CSV 缺少 payload_byte_* 列"
    assert "label" in df.columns, "CSV 缺少 label 列"

    X = df[payload_cols].fillna(0).to_numpy(dtype=np.int64)
    # 0-255
    X = np.clip(X, 0, 255)
    # 标签转 int
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

    # 70/20/10
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_bal, y_bal, test_size=0.30, random_state=SEED, stratify=y_bal)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=2/3, random_state=SEED, stratify=y_tmp)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ========= transformer-based model =========
def build_model():
    config = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=MAX_LEN,
        num_labels=NUM_LABELS,
        pad_token_id=0,
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = BertForSequenceClassification(config)
    return model

# ========= eval =========
@torch.no_grad()
def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, digits=4)
    return acc, f1, report
@torch.no_grad()
def benchmark_inference(model, loader, device, desc="Benchmark(Test)"):
    model.eval()
    n_items = 0
    model_forward_time = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        bs = input_ids.size(0)
        n_items += bs

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_iter0 = time.perf_counter()

        # 纯前向（不传 labels，避免计算 loss）
        with torch.inference_mode():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        if device.type == "cuda":
            torch.cuda.synchronize()
        model_forward_time += (time.perf_counter() - t_iter0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    overall_ms_per_item = (t1 - t0) * 1000.0 / max(1, n_items)
    model_ms_per_item   = model_forward_time * 1000.0 / max(1, n_items)
    return overall_ms_per_item, model_ms_per_item

# ========= main to train =========
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_and_prepare(CSV_PATH)
    print(f"Balanced sizes -> train:{len(X_tr)} val:{len(X_val)} test:{len(X_te)}")

    train_ds = ByteDataset(X_tr, y_tr)
    val_ds   = ByteDataset(X_val, y_val)
    test_ds  = ByteDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model().to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_f1 = -1.0
    best_path = os.path.join(SAVE_DIR, "best.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} (train)"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 可选，稳定训练
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        val_acc, val_f1, _ = evaluate(model, val_loader, device, desc="Valid")
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={running_loss/len(train_loader):.4f} "
              f"| val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model_state": model.state_dict(),
                        "config": model.config.to_dict()}, best_path)
            print(f"  -> saved best to {best_path}")

    ckpt = torch.load(best_path, map_location="cpu")
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    # —— 评估 ——
    te_acc, te_f1, te_report = evaluate(model, test_loader, device, desc="Test")
    print("\n=== Test ===")
    print(f"Acc: {te_acc:.4f} | F1(weighted): {te_f1:.4f}")
    print(te_report)
    # —— 时间 ——
    overall_ms, model_ms = benchmark_inference(model, test_loader, device, desc="Benchmark(Test)")
    print("\n=== Inference Benchmark (per item) ===")
    print(f"End-to-end: {overall_ms:.3f} ms/item | Throughput: {1000.0/overall_ms:.2f} items/s")
    print(f"Model-only: {model_ms:.3f} ms/item   | Throughput: {1000.0/model_ms:.2f} items/s")


if __name__ == "__main__":
    main()
