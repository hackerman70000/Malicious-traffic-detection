import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data():
    data = pd.read_csv("data/raw/UNSW-NB15/Data.csv")
    labels = pd.read_csv("data/raw/UNSW-NB15/Label.csv")

    df = pd.merge(data, labels, left_index=True, right_index=True)

    X = df.drop("Label", axis=1)
    y = df["Label"]

    # Binary classification data
    y_binary = (y > 0).astype(int)

    # Filter data for multiclass (only attacks)
    attack_mask = y > 0
    X_attacks = X[attack_mask]
    y_multiclass = y[attack_mask]

    # Adjust multiclass labels to start from 0 (since we removed benign class)
    y_multiclass = y_multiclass - 1

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        X_attacks[col] = le.fit_transform(X_attacks[col])

    return X, y_binary, X_attacks, y_multiclass


def create_directories():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    binary_dir = os.path.join("models", "trained", "binary", f"xgboost_{timestamp}")
    multiclass_dir = os.path.join(
        "models", "trained", "multiclass", f"xgboost_{timestamp}"
    )
    os.makedirs(binary_dir, exist_ok=True)
    os.makedirs(multiclass_dir, exist_ok=True)
    return binary_dir, multiclass_dir


def plot_confusion_matrix(cm, labels, title, output_path):
    plt.figure(figsize=(10, 8))

    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        annot_kws={"size": 12},
        cbar_kws={"shrink": 0.8},
    )

    plt.title(title, pad=20, size=14, weight="bold")

    plt.ylabel("True Label", size=12, labelpad=15)
    plt.xlabel("Predicted Label", size=12, labelpad=15)

    plt.xticks(rotation=45, ha="right", size=10)
    plt.yticks(rotation=0, va="center", size=10)

    plt.tight_layout(pad=1.5)

    plt.savefig(output_path, bbox_inches="tight", dpi=300, pad_inches=0.5)
    plt.close()


def train_and_evaluate():
    X, y_binary, X_attacks, y_multiclass = load_and_preprocess_data()
    binary_dir, multiclass_dir = create_directories()

    # Binary Classification
    X_train, X_test, y_binary_train, y_binary_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    binary_model = xgb.XGBClassifier(
        objective="binary:logistic", random_state=42, n_estimators=100
    )
    binary_model.fit(X_train, y_binary_train)
    binary_pred = binary_model.predict(X_test)

    binary_cm = confusion_matrix(y_binary_test, binary_pred)
    binary_report = classification_report(
        y_binary_test, binary_pred, target_names=["Benign", "Malicious"]
    )

    plot_confusion_matrix(
        binary_cm,
        ["Benign", "Malicious"],
        "Binary Classification Confusion Matrix",
        os.path.join(binary_dir, "confusion_matrix.png"),
    )

    binary_model.save_model(os.path.join(binary_dir, "model.json"))

    with open(os.path.join(binary_dir, "report.txt"), "w") as f:
        f.write("XGBoost Binary Classification Results\n")
        f.write("==================================\n\n")
        f.write(binary_report)

    # Multiclass Classification (only attacks)
    X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
        X_attacks, y_multiclass, test_size=0.2, random_state=42
    )

    multi_model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=9,
        random_state=42,
        n_estimators=100,
    )
    multi_model.fit(X_multi_train, y_multi_train)
    multi_pred = multi_model.predict(X_multi_test)

    multi_cm = confusion_matrix(y_multi_test, multi_pred)
    multi_report = classification_report(
        y_multi_test,
        multi_pred,
        target_names=[
            "Analysis",
            "Backdoor",
            "DoS",
            "Exploits",
            "Fuzzers",
            "Generic",
            "Reconnaissance",
            "Shellcode",
            "Worms",
        ],
    )

    plot_confusion_matrix(
        multi_cm,
        [
            "Analysis",
            "Backdoor",
            "DoS",
            "Exploits",
            "Fuzzers",
            "Generic",
            "Reconnaissance",
            "Shellcode",
            "Worms",
        ],
        "Multiclass Classification Confusion Matrix (Attacks Only)",
        os.path.join(multiclass_dir, "confusion_matrix.png"),
    )

    multi_model.save_model(os.path.join(multiclass_dir, "model.json"))

    with open(os.path.join(multiclass_dir, "report.txt"), "w") as f:
        f.write("XGBoost Multiclass Classification Results (Attacks Only)\n")
        f.write("================================================\n\n")
        f.write(multi_report)

    return binary_dir, multiclass_dir


if __name__ == "__main__":
    binary_dir, multiclass_dir = train_and_evaluate()
    print(f"Binary classification results saved in: {binary_dir}")
    print(f"Multiclass classification results saved in: {multiclass_dir}")
