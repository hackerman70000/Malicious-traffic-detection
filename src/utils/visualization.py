from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray, labels: List[str], title: str, output_path: Path
) -> None:
    """Plot and save confusion matrix."""
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
