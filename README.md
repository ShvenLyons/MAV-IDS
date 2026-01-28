# Basic IDS Experiments

This repository contains six main scripts for packet-level intrusion detection based on byte-level payloads:

1. `word2vec_RF.py` – Word2Vec + Random Forest baseline  
2. `CompareVerify.py` – Compare trained Word2Vec vs random Word2Vec with the same RF  
3. `baseline_ml.py` – Classic machine learning baselines (DT / KNN / GBM / LR)  
4. `baseline_dl.py` – Deep learning baselines (DNN / CNN / RNN / LSTM / 1D-CNN / 2D-CNN)  
5. `transformer_based.py` – Byte-level Transformer (BERT-style)  
6. `parallel_vit.py` – Byte-to-image ViT / Parallel ViT models  

---

## 1. Environment

Recommended:

- Python ≥ 3.10  
- PyTorch (GPU if available)  
- Common ML/ DL libraries: `numpy`, `pandas`, `scikit-learn`, `gensim`, `transformers`, `tqdm`, etc.

If you have `requirements.txt`:

```bash
pip install -r requirements.txt

```

## 2. Data

All scripts assume preprocessed CSV files with:

- 1500 payload-byte features (e.g. `payload_byte_1` … `payload_byte_1500` or similar)
- A label column (binary classification)

Typical default paths inside scripts (you can change them in each `.py` file):

- CICIDS-2017:  
  `data/CICIDS/Payload_data_CICIDS2017_train_split_binary.csv`
  `data/CICIDS/Payload_data_CICIDS2017_test_split_binary.csv`  

- UNSW-NB15:  
  `data/UNSW/Payload_data_UNSW_train_split_binary.csv`
  `data/UNSW/Payload_data_UNSW_test_split_binary.csv`  

- MAVLink:  
  `data/MAVLink/train.csv`  
  `data/MAVLink/test.csv`  

Please adjust the paths in the scripts to match your local data layout.

## 3. word2vec_RF.py

**Aim**

Train a Word2Vec embedding on byte-level payloads and a Random Forest classifier on the resulting packet embeddings.

**RunCommand**

1. Open `word2vec_RF.py` and set the paths at the top of the file, for example:

   - `train_path` (train CSV)
   - `test_path` (optional test CSV)
   - `w2v_model_path` (e.g. `model/RF/XXX.model`)
   - `rf_model_path` (e.g. `model/RF/XXX.pkl`)

2. Run in the project root:

   ```bash
   python word2vec_RF.py
   ```
   
## 4. `CompareVerify.py`

**Aim**

Compare a trained Word2Vec + RF pipeline with a random-initialized Word2Vec using the same RF classifier on test data.

**RunCommand**

1. Open `CompareVerify.py` and ensure `model_configs` points to existing models and test CSVs, for example:

   - `test_path` (test CSV for each dataset)
   - `w2v_model_path` (e.g. `model/RF/UNSW.model`, `model/RF/CIC.model`, `model/RF/MAV.model`, etc.)
   - `rf_model_path` (e.g. `model/RF/UNSW.pkl`, `model/RF/CIC.pkl`, `model/RF/MAV.pkl`, etc.)

2. Run in the project root:

   ```bash
   python CompareVerify.py
   ```

## 5. `baseline_ml.py`

**Aim**

Provide classic machine learning baselines (Decision Tree, KNN, Gradient Boosting, Logistic Regression) on byte-level payload features.

**Model Choose**

The model is selected by the command-line argument `model_type` (positional or `--model_type`).

Available options typically include:

- `dt` → Decision Tree  
- `knn` → K-Nearest Neighbors  
- `gbm` → Gradient Boosting  
- `lr` → Logistic Regression  

Make sure `CSV_PATH` inside `baseline_ml.py` points to your training CSV (e.g. CICIDS payload data).

**RunCommand**

From the project root:

**Decision Tree**

```bash
python baseline_ml.py dt
# or
python baseline_ml.py --model_type dt
```

**KNN**

```bash
python baseline_ml.py knn
# or
python baseline_ml.py --model_type knn
```

**Gradient Boosting**

```bash
python baseline_ml.py gbm
# or
python baseline_ml.py --model_type gbm
```

**Logistic Regression**

```bash
python baseline_ml.py lr
# or
python baseline_ml.py --model_type lr
```

## 6. baseline_dl.py
**Aim**

Provide deep learning baselines (DNN, CNN, RNN, LSTM, etc.) on byte-level payload features with configurable architectures.

**Model Choose**

The model is selected by the command-line argument `model_type` (positional or `--model_type`).

Typical options include (depending on your implementation):

- `dnn` → Fully-connected neural network  
- `cnn` → 1D CNN baseline  
- `rnn` → Simple RNN baseline  
- `lstm` → LSTM baseline  
- `cnn1d` → Alternative 1D CNN  
- `cnn2d` → 2D CNN (reshape bytes into a 2D map)  

Make sure `CSV_PATH` inside `baseline_dl.py` points to your training CSV (e.g. CICIDS payload data).

**Model Change**

You can change CNN and LSTM hyperparameters directly inside the function build_model in baseline_dl.py. For example, the current version may look like
```python
def build_model(model_type):
    if model_type == "cnn":
        return CNN(
            input_len=MAX_LEN,
            num_labels=NUM_LABELS,
            num_convs=2,
            base_channels=32,
            channel_mult=2
        )
    elif model_type == "lstm":
        return LSTM(
            hidden=HIDDEN_SIZE,
            num_labels=NUM_LABELS,
            seq_len=1500,
            input_size=1
        )
```
You can modify these arguments to adjust the architecture.


**RunCommand**

DNN
```bash
python baseline_dl.py dnn
# or
python baseline_dl.py --model_type dnn
```

CNN
```bash
python baseline_dl.py cnn
# or
python baseline_dl.py --model_type cnn
```

RNN
```bash
python baseline_dl.py rnn
# or
python baseline_dl.py --model_type rnn
```

LSTM
```bash
python baseline_dl.py lstm
# or
python baseline_dl.py --model_type lstm
```

1D-CNN
```bash
python baseline_dl.py cnn1d
# or
python baseline_dl.py --model_type cnn1d
```

2D-CNN
```bash

python baseline_dl.py cnn2d

or python baseline_dl.py --model_type cnn2d
```

## 8. `transformer_based.py`

**Aim**

Train a byte-level Transformer (BERT-style) for payload-based intrusion detection by treating each byte as a token.

**RunCommand**

1. Ensure `CSV_PATH` in `transformer_based.py` points to the desired training CSV.
2. Adjust model hyperparameters at the top of the file if needed (hidden size, number of layers, etc.).
3. Run in the project root:

   ```bash
   python transformer_based.py
   ```

## 9. `parallel_vit.py`

**Aim**

Convert byte-level payloads into image-like representations and train a ViT or Parallel ViT model for intrusion detection.

**RunCommand**

1. Open `parallel_vit.py` and check the key configuration at the top:

   - `MODEL_TYPE` (use `"vit"` for standard ViT or `"pvit"` for Parallel ViT)
   - `PVIT_BRANCH` (number of branches when using Parallel ViT)
   - `CSV_PATH` (payload CSV)
   - `SAVE_DIR` (directory to save checkpoints)

2. To run **standard ViT**:

   - Set:

     ```python
     MODEL_TYPE = "vit"
     ```

   - Then run from the project root:

     ```bash
     python parallel_vit.py
     ```

3. To run **Parallel ViT**:

   - Set:

     ```python
     MODEL_TYPE = "pvit"
     # and adjust
     PVIT_BRANCH = <number_of_branches_you_want>
     ```

   - Then run from the project root:

     ```bash
     python parallel_vit.py
     ```
