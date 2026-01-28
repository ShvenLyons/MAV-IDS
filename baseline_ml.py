import os, time, random
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# ========== Config ==========
SEED = 42
MAX_LEN = 1500
CSV_PATH = "data/CICIDS/Payload_data_CICIDS2017_train_split_binary.csv"
SAVE_DIR = "model/checkpoints/ml_baselines"


# ========== Utils ==========
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classic ML baselines (DT/KNN/GBM/LR) for byte payloads"
    )
    parser.add_argument(
        "model_type",
        nargs="?",
        default="dt",
        choices=["dt", "knn", "gbm", "lr"],
        help="Model type to use (default: dt)",
    )

    parser.add_argument(
        "--model_type",
        dest="model_type_kw",
        choices=["dt", "knn", "gbm", "lr"],
        help="Model type to use (overrides positional if provided)",
    )

    args = parser.parse_args()
    return args.model_type_kw if args.model_type_kw is not None else args.model_type


# ========== Data load & preprocess ==========
def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)
    payload_cols = [f"payload_byte_{i}" for i in range(1, MAX_LEN + 1)]
    assert all(c in df.columns for c in payload_cols), "缺少 payload_byte_* 列"
    assert "label" in df.columns, "缺少 label 列"

    X = df[payload_cols].fillna(0).to_numpy(dtype=np.float32)
    X = np.clip(X, 0, 255) / 255.0
    y = df["label"].astype(int).to_numpy()

    unique, counts = np.unique(y, return_counts=True)
    min_n = counts.min()
    idxs_bal = np.concatenate(
        [
            np.random.choice(np.where(y == cls)[0], size=min_n, replace=False)
            for cls in unique
        ]
    )
    np.random.shuffle(idxs_bal)
    X_bal, y_bal = X[idxs_bal], y[idxs_bal]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_bal, y_bal, test_size=0.30, random_state=SEED, stratify=y_bal
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=2 / 3, random_state=SEED, stratify=y_tmp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    

# ========== Models ==========
def build_model(model_type: str):
    model_type = model_type.lower()
    if model_type == "dt":
        return DecisionTreeClassifier(random_state=SEED)
    elif model_type == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    elif model_type == "gbm":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED)
    elif model_type == "lr":
        return LogisticRegression(max_iter=1000, n_jobs=-1, random_state=SEED)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ========== Eval ==========
def evaluate_split(model, X, y, split_name: str):
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, digits=4)

    print(f"\n=== {split_name} ===")
    print(
        f"Acc: {acc:.4f}  Prec(w): {prec:.4f}  Rec(w): {rec:.4f}  F1(w): {f1:.4f}"
    )
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    return acc, prec, rec, f1


def predict_inference(model, X_test, batch_size: int = 8192):
    n = X_test.shape[0]
    if n == 0:
        return 0.0
    t0 = time.perf_counter()
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        _ = model.predict(X_test[start:end])
        start = end
    t1 = time.perf_counter()
    ms_per_item = (t1 - t0) * 1000.0 / n
    return ms_per_item


# ========== Main ==========
def main():
    set_seed(SEED)
    model_type = parse_args()
    print(f"Using model type: {model_type}")

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_and_prepare(CSV_PATH)
    print(f"train:{len(X_tr)} val:{len(X_va)} test:{len(X_te)}")

    model = build_model(model_type)
    print("\nModel:", model)

    print("\nTraining...")
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    t1 = time.perf_counter()
    print(f"Training finished in {(t1 - t0):.2f} seconds.")

    evaluate_split(model, X_te, y_te, "Test")
    # ms_per_item = predict_inference(model, X_te)
    # print("===  Prediiction ===")
    # print(
    #     f"{model_type.upper()} | {ms_per_item:.3f} ms/item | "
    #     f"Throughput: {1000.0/ms_per_item:.2f} items/s"
    # )

    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, f"baseline_ml_{model_type}.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
