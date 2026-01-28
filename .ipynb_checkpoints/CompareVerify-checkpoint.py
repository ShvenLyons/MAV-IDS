import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

def complete_payload_columns(df):
    expected_cols = [f'payload_byte_{i}' for i in range(1, 1501)]
    existing_payload_cols = [col for col in df.columns if col.startswith("payload_byte_")]
    print(f"Found {len(existing_payload_cols)} payload columns in dataset")
    print(f"Expected 1500 payload columns, will pad with zeros")
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    all_cols = expected_cols + ['label']
    df = df[all_cols]
    return df, expected_cols

def load_and_test_models(dataset_name, test_path, w2v_model_path, rf_model_path, sample_size=10000, needs_padding=False):
    """
    Parameters:
    - dataset_name:
    - test_path
    - w2v_model_path
    - rf_model_path
    - sample_size
    - needs_padding
    """
    print(f"\n{'='*50}")
    print(f"DATASET {dataset_name} EXP")
    print(f"{'='*50}")
    
    # ========== File Check ==========
    if not os.path.exists(test_path):
        print(f"Test dataset doesn't exist in {test_path}")
        return None, None
    if not os.path.exists(w2v_model_path):
        print(f"Word2Vec model file doesn't exist in {w2v_model_path}")
        return None, None
    if not os.path.exists(rf_model_path):
        print(f"RF model file doesn't exist in{rf_model_path}")
        return None, None
    
    # ========== 加载测试数据 ==========
    print("loading test data...")
    df_test_full = pd.read_csv(test_path)
    
    # ========== 分层抽样 ==========
    def stratified_sample(df, label_col, sample_size):
        grouped = df.groupby(label_col, group_keys=False)
        total_size = len(df)
        frac = sample_size / total_size
        return grouped.apply(lambda x: x.sample(
            frac=frac,
            replace=(frac > 1),
            random_state=24
        ))
    
    df_test = stratified_sample(df_test_full, 'label', sample_size)
    print(f"test data: {len(df_test)} packets")
    print(df_test['label'].value_counts(normalize=True))
    
    # ========== MAVLink: payload padding ==========
    if needs_padding:
        print("MAVLink payload padding...")
        df_test, feature_cols = complete_payload_columns(df_test)
        print(f"payload reading")
    else:
        feature_cols = [col for col in df_test.columns if col.startswith("payload_byte_")]
        if len(feature_cols) != 1500:
            raise ValueError("payload byte number != 1500")
        print(f"payload reading")
    
    # ========== 转换为token序列 ==========
    tokens_test = df_test[feature_cols].astype(str).values.tolist()
    print("token embedding")
    
    # ========== 加载Word2Vec模型 ==========
    print("Loading Word2Vec Model...")
    w2v = Word2Vec.load(w2v_model_path)
    print(f"From {w2v_model_path}")
    
    # ========== 嵌入函数 ==========
    def embed(tokens):
        vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size)
    
    # ========== 生成嵌入向量 ==========
    test_embeds = np.vstack([embed(seq) for seq in tokens_test])
    
    # ========== 加载RF模型 ==========
    print("Loading RF Model...")
    rf = joblib.load(rf_model_path)
    print(f"From {rf_model_path}")
    
    # ========== 模型预测 ==========
    print("Predicting...")
    y_test = df_test['label'].values
    y_pred = rf.predict(test_embeds)
    
    # ========== 评估结果 ==========
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    
    print(f"\n{dataset_name} Result:")
    print(f"Acc: {acc:.4f}")
    print("Report:")
    print(report)
    
    return acc, report

def test_with_blank_w2v(dataset_name, test_path, rf_model_path, sample_size=10000, needs_padding=False):

    print(f"\n{'='*50}")
    print(f"Blank Word2Vec - {dataset_name} Dataset")
    print(f"{'='*50}")
    
    # ========== 检查文件是否存在 ==========
    if not os.path.exists(test_path):
        print(f"Test dataset doesn't exist in {test_path}")
        return None, None
    if not os.path.exists(rf_model_path):
        print(f"RF model file doesn't exist in {rf_model_path}")
        return None, None
    
    # ========== 加载测试数据 ==========
    print("loading test data...")
    df_test_full = pd.read_csv(test_path)
    
    # ========== 分层抽样 ==========
    def stratified_sample(df, label_col, sample_size):
        grouped = df.groupby(label_col, group_keys=False)
        total_size = len(df)
        frac = sample_size / total_size
        return grouped.apply(lambda x: x.sample(
            frac=frac,
            replace=(frac > 1),
            random_state=24
        ))
    
    df_test = stratified_sample(df_test_full, 'label', sample_size)
    print(f"Test dataset: {len(df_test)} packets")
    print(df_test['label'].value_counts(normalize=True))
    
    # ========== MAVLink payload padding ==========
    if needs_padding:
        print("MAVLink payload padding...")
        df_test, feature_cols = complete_payload_columns(df_test)
        print(f" payload reading")
    else:
        feature_cols = [col for col in df_test.columns if col.startswith("payload_byte_")]
        if len(feature_cols) != 1500:
            raise ValueError(" payload byte number != 1500")
        print(f" payload reading")
    
    # ========== 转换为token序列 ==========
    tokens_test = df_test[feature_cols].astype(str).values.tolist()
    print(f" token embedding")
    
    # ========== 创建空白Word2Vec模型 ==========
    print("Building Blank Word2Vec...")
    # 创建一个未经训练的Word2Vec模型，结构与原模型相同
    blank_w2v = Word2Vec(vector_size=32, window=3, min_count=1, workers=1)
    
    # 构建词汇表（使用测试集中的所有token）
    all_tokens = set()
    for seq in tokens_test:
        all_tokens.update(seq)
    
    # 构建一个简单的词汇表
    sentences = [[token] for token in all_tokens]
    blank_w2v.build_vocab(sentences)
    
    # 随机初始化权重
    blank_w2v.wv.vectors = np.random.normal(0, 0.1, blank_w2v.wv.vectors.shape)
    print("Blank Word2Vec Built")
    
    # ========== 嵌入函数 ==========
    def embed_blank(tokens):
        vecs = [blank_w2v.wv[t] for t in tokens if t in blank_w2v.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(blank_w2v.vector_size)
    
    # ========== 生成嵌入向量 ==========
    print("Blank Word2Vec embbedding...")
    test_embeds = np.vstack([embed_blank(seq) for seq in tokens_test])
    print(f"Vector Shape: {test_embeds.shape}")
    
    # ========== 加载RF模型 ==========
    print("Loading RF Model...")
    rf = joblib.load(rf_model_path)
    print(f"From {rf_model_path}")
    
    # ========== 模型预测 ==========
    print("Predicting...")
    y_test = df_test['label'].values
    y_pred = rf.predict(test_embeds)
    
    # ========== 评估结果 ==========
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    
    print(f"\n{dataset_name} Result - Blank Word2Vec EXP")
    print(f"Acc: {acc:.4f}")
    print("Report:")
    print(report)
    
    return acc, report

# ========== 定义三组模型的路径 ==========
model_configs = [
    {
        "name": "UNSW",
        "test_path": "./data/UNSW/Payload_data_UNSW_test_split_binary.csv",
        "w2v_model_path": "model/RF/UNSW.model",
        "rf_model_path": "model/RF/UNSW.pkl",
        "needs_padding": False
    },
    {
        "name": "CICIDS", 
        "test_path": "./data/CICIDS/Payload_data_CICIDS2017_test_split_binary.csv",
        "w2v_model_path": "model/RF/CIC.model",
        "rf_model_path": "model/RF/CIC.pkl",
        "needs_padding": False
    },
    {
        "name": "MAVLink",
        "test_path": "./data/MAVLink/test.csv", 
        "w2v_model_path": "model/RF/MAV.model",
        "rf_model_path": "model/RF/MAV.pkl",
        "needs_padding": True  
    }
]

# ========== 正常验证: 使用训练好的Word2Vec和RF模型 ==========
print("Start trained-W2V + trained-RF")
normal_results = {}

for config in model_configs:
    try:
        acc, report = load_and_test_models(
            dataset_name=config["name"],
            test_path=config["test_path"],
            w2v_model_path=config["w2v_model_path"],
            rf_model_path=config["rf_model_path"],
            sample_size=10000,
            needs_padding=config["needs_padding"]
        )
        normal_results[config["name"]] = {
            "accuracy": acc,
            "report": report
        }
    except Exception as e:
        print(f"Error {config['name']} when: {str(e)}")
        normal_results[config["name"]] = None

# ========== RF单独验证: 使用空白Word2Vec + 预训练RF ==========
print("\n\nStart Blank-W2V + Fixed-RF")
blank_w2v_results = {}

for config in model_configs:
    try:
        acc, report = test_with_blank_w2v(
            dataset_name=config["name"],
            test_path=config["test_path"],
            rf_model_path=config["rf_model_path"],
            sample_size=10000,
            needs_padding=config["needs_padding"]
        )
        blank_w2v_results[config["name"]] = {
            "accuracy": acc,
            "report": report
        }
    except Exception as e:
        print(f"验证 {config['name']} 时出现错误: {str(e)}")
        blank_w2v_results[config["name"]] = None

# ========== 汇总结果 ==========
print(f"\n{'='*80}")
print("Total Result")
print(f"{'='*80}")

print("\nTrained-W2V + Trained-RF")
for name, result in normal_results.items():
    if result is not None:
        print(f"  {name}: Acc = {result['accuracy']:.4f}")
    else:
        print(f"  {name} Failed")

print("\nBlank-W2V + Fixed-RF:")
for name, result in blank_w2v_results.items():
    if result is not None:
        print(f"  {name}: Acc = {result['accuracy']:.4f}")
    else:
        print(f"  {name} Failed")

print(f"\n{'='*80}")
print("Compare")
print(f"{'='*80}")

for name in normal_results.keys():
    if normal_results.get(name) and blank_w2v_results.get(name):
        normal_acc = normal_results[name]["accuracy"]
        blank_acc = blank_w2v_results[name]["accuracy"]
        diff = normal_acc - blank_acc
        print(f"{name}:")
        print(f"  NormalAcc: {normal_acc:.4f}")
        print(f"  BlankAcc: {blank_acc:.4f}")
        print(f"  LOWER: {diff:.4f} ({diff*100:.2f}%)")