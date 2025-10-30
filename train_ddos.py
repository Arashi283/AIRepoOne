import pandas as pd
import numpy as np
import os, zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from tqdm import tqdm
import joblib

# === CONFIG ===
DATASET_ZIP = r"C:\Users\MOHD MUZAMMIL\Downloads\archive (4).zip"
EXTRACT_PATH = r"C:\Users\MOHD MUZAMMIL\VSCode\DDos\data"
MODEL_PATH = "best_ddos_model.pkl"

# === EXTRACT DATASET ===
print(f"Extracting dataset from {DATASET_ZIP} ...")
with zipfile.ZipFile(DATASET_ZIP, "r") as z:
    z.extractall(EXTRACT_PATH)

print("📂 Loading parquet files...")
all_files = [os.path.join(EXTRACT_PATH, f) for f in os.listdir(EXTRACT_PATH) if f.endswith(".parquet")]

df = pd.concat([pd.read_parquet(f) for f in tqdm(all_files)], ignore_index=True)
df = df.dropna().loc[:, ~df.columns.duplicated()]

le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])

X = df.select_dtypes(include=[np.number])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# === TRAIN MODEL ONCE ===
model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=80,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1
)

print("🚀 Training model...")
model.fit(X_train, y_train)

# === EVALUATE ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred))

# === SAVE MODEL & ENCODER ===
joblib.dump(model, MODEL_PATH)
joblib.dump(le, "label_encoder.pkl")
print(f"💾 Model saved as: {MODEL_PATH}")
print(f"💾 Label encoder saved as: label_encoder.pkl")
