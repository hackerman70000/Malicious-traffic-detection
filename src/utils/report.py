from typing import List

import numpy as np


def create_confusion_matrix_text(cm: np.ndarray, labels: List[str]) -> str:
    """
    Create text representation of confusion matrix.

    Args:
        cm: Confusion matrix array
        labels: List of class labels (Benign, Malicious)

    Returns:
        Formatted string representation of confusion matrix
    """
    cm_text = "Confusion Matrix:\n"
    cm_text += "-" * 60 + "\n"

    header = "Predicted →"
    cm_text += " " * 15 + header + "\n"
    cm_text += " " * 15 + "".join(f"{label:>12}" for label in labels) + "\n"
    cm_text += " " * 15 + "-" * (12 * len(labels)) + "\n"

    for i, label in enumerate(labels):
        cm_text += f"Actual {label:>8} |"
        cm_text += "".join(f"{cm[i, j]:>12}" for j in range(len(labels))) + "\n"

    return cm_text + "\n"


def create_classification_report_text(
    precision_report: dict, cm_text: str, detailed_report: str
) -> str:
    """
    Create complete binary classification report text.

    Args:
        precision_report: Dictionary containing precision metrics
        cm_text: Text representation of confusion matrix
        detailed_report: Detailed classification report

    Returns:
        Complete formatted report text
    """
    report = []

    report.append("XGBoost Binary Classification Results")
    report.append("=" * 35)
    report.append("")

    report.append("Performance Metrics:")
    report.append(
        f"Malicious class precision: {precision_report['Malicious']['precision']:.2%}"
    )
    report.append(
        f"Benign class precision: {precision_report['Benign']['precision']:.2%}"
    )
    report.append(
        f"Overall precision: {precision_report['macro avg']['precision']:.2%}"
    )
    report.append("")

    report.append(cm_text)

    report.append("Detailed Classification Report:")
    report.append(detailed_report)

    return "\n".join(report)
