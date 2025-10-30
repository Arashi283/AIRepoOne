import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import time


# === Step 1: Select file ===
print("📂 Select your dataset file to test the model...")
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select your dataset file",
    filetypes=[
    ("Supported files", "*.csv *.xlsx *.xls *.parquet *.pcap_ISCX"),
    ("All files", "*.*")
]

)

if not file_path:
    print("❌ No file selected, exiting.")
    exit()

print(f"✅ File selected: {file_path}")

# === Step 2: Load model & encoder ===
print("✅ Loading model and encoder...")
model = joblib.load("best_ddos_model.pkl")
le = joblib.load("label_encoder.pkl")

# === Step 3: Load dataset (auto-detect file type) ===
print("📥 Loading dataset...")
if file_path.endswith(".parquet"):
    X = pd.read_parquet(file_path)
elif file_path.endswith(".csv"):
    X = pd.read_csv(file_path)
elif file_path.endswith((".xls", ".xlsx")):
    X = pd.read_excel(file_path)
else:
    print("❌ Unsupported file type. Please upload a .csv, .parquet, or .xlsx file.")
    exit()

print(f"📊 Loaded dataset shape: {X.shape}")


# Convert all columns to numeric if possible
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
X = X.fillna(0)

# === Step 4: Feature alignment ===
if hasattr(model, "feature_name_"):
    model_features = list(model.feature_name_)
else:
    try:
        booster = getattr(model, "booster_", None)
        if booster is not None and hasattr(booster, "feature_name"):
            model_features = list(booster.feature_name())
        else:
            model_features = list(X.columns)
    except Exception:
        model_features = list(X.columns)

missing_cols = [f for f in model_features if f not in X.columns]
if missing_cols:
    print(f"Adding {len(missing_cols)} missing features with zeros: {missing_cols[:5]}")
    for c in missing_cols:
        X[c] = 0.0

extra_cols = [c for c in X.columns if c not in model_features]
if extra_cols:
    print(f"Dropping {len(extra_cols)} extra features: {extra_cols[:5]}")
    X = X.drop(columns=extra_cols, errors='ignore')

X = X[model_features]
X = X.astype(float)  # ✅ Force all numeric to avoid category mismatch

print(f"✅ Aligned test data shape: {X.shape}")

# === Step 5: Predict safely ===
print("🧠 Making predictions...")
try:
    preds = model.predict(X, raw_score=False)
except Exception as e:
    print(f"⚠️ Model prediction issue detected: {e}")
    print("🔧 Retrying with .values to bypass pandas metadata...")
    preds = model.predict(X.values)

labels = le.inverse_transform(
    (preds > 0.5).astype(int) if preds.ndim > 1 else preds.astype(int)
)

print("\n✅ Predictions complete!")
print(f"🧾 Predicted label distribution:\n{pd.Series(labels).value_counts()}")


# === Step 6: SHAP Explainability (clean visualization) ===
print("\n🔍 Generating SHAP explanations (this may take a moment)...")

sample = X.sample(min(300, len(X)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# Create unique file names (timestamped)
timestamp = time.strftime("%Y%m%d_%H%M%S")
summary_path = f"shap_summary_{timestamp}.png"
bar_path = f"shap_bar_{timestamp}.png"

# --- SHAP summary (dot plot) ---
plt.figure(figsize=(14, 8))
shap.summary_plot(shap_values, sample, show=False)
plt.title("SHAP Summary Plot (Feature Impact on Model Output)", fontsize=14)
plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.close()
print(f"✅ SHAP summary plot saved as: {summary_path}")

# --- SHAP bar plot (average importance) ---
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
plt.title("Top Features by Average SHAP Importance", fontsize=14)
plt.tight_layout()
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"✅ SHAP bar plot saved as: {bar_path}")

print("\n🎯 Done! You can open the new SHAP plots for this run.")
