"""
Test script for Fraud Dataset Benchmark (FDB) - ieeecis dataset
"""
import sys
import os

# Ensure kaggle CLI path is available
kaggle_bin = os.path.expanduser("~/Library/Python/3.13/bin")
if kaggle_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{kaggle_bin}:{os.environ.get('PATH', '')}"

from fdb.datasets import FraudDatasetBenchmark
import numpy as np
import pandas as pd

print("=" * 60)
print("FDB Test - IEEE-CIS Fraud Detection Dataset")
print("=" * 60)

# Load the ieeecis dataset
# use load_pre_downloaded=True on subsequent runs to avoid re-downloading
print("\nLoading ieeecis dataset from Kaggle...")
print("(First run will download ~500MB, this may take a few minutes)\n")

obj = FraudDatasetBenchmark(
    key="ieeecis",
    load_pre_downloaded=False,
    delete_downloaded=True,
)

print(f"Dataset key: {obj.key}")
print(f"\nTrain set shape: {obj.train.shape}")
print(f"Test set shape:  {obj.test.shape}")
print(f"Test labels shape: {obj.test_labels.shape}")

print(f"\nTrain columns ({len(obj.train.columns)}):")
print(obj.train.columns.tolist())

print(f"\nFirst 5 rows of train:")
print(obj.train.head())

print(f"\nLabel distribution (train):")
print(obj.train["EVENT_LABEL"].value_counts())
print(f"\nFraud rate (train): {obj.train['EVENT_LABEL'].mean():.4f}")

print(f"\nLabel distribution (test):")
print(obj.test_labels["EVENT_LABEL"].value_counts())
print(f"Fraud rate (test): {obj.test_labels['EVENT_LABEL'].mean():.4f}")

# Test eval with dummy predictions
print("\n" + "=" * 60)
print("Testing eval() with dummy predictions")
print("=" * 60)
dummy_preds = np.random.rand(len(obj.test_labels))
metrics = obj.eval(dummy_preds)
print(f"AUC-ROC:  {metrics['roc_score']:.4f}")
print(f"TPR@1%FPR: {metrics['tpr_1fpr']:.4f}")

print("\nSetup complete! FDB is working correctly.")
