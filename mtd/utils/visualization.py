import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb


def plot_confusion_matrix(
    cm: np.ndarray, labels: List[str], title: str, output_path: Path
) -> None:
    """Plot and save a confusion matrix with enhanced styling.

    Args:
        cm: Confusion matrix array
        labels: List of class labels
        title: Title for the plot
        output_path: Path where the plot should be saved
    """
    try:
        if not isinstance(cm, np.ndarray):
            raise ValueError("Confusion matrix must be a numpy array")
        if (
            len(cm.shape) != 2
            or cm.shape[0] != len(labels)
            or cm.shape[1] != len(labels)
        ):
            raise ValueError("Confusion matrix dimensions must match number of labels")

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
        logging.info(f"Successfully saved confusion matrix plot to {output_path}")

    except Exception as e:
        logging.error(f"Error plotting confusion matrix: {str(e)}")
        plt.close()


def plot_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: List[str],
    output_path: Path,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Plot feature importance scores from the model.

    Args:
        model: Trained XGBoost classifier
        feature_names: List of feature names
        output_path: Path where the plot should be saved
        top_n: Number of top features to display
        figsize: Figure size (width, height)
    """
    try:
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not have feature importance scores")

        importance_scores = model.feature_importances_
        if importance_scores is None or len(importance_scores) == 0:
            raise ValueError("Feature importance scores are empty")

        if len(importance_scores) != len(feature_names):
            raise ValueError("Number of features does not match importance scores")

        importance_dict = dict(zip(feature_names, importance_scores))
        sorted_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        features, scores = zip(*sorted_features)

        plt.figure(figsize=figsize)

        bars = plt.barh(range(len(scores)), scores)
        plt.yticks(range(len(features)), features)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, i, f"{width:.4f}", ha="left", va="center", fontsize=10)

        plt.xlabel("Importance Score", size=12, labelpad=10)
        plt.title(
            f"Top {top_n} Most Important Features", size=14, pad=20, weight="bold"
        )
        plt.grid(axis="x", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"Successfully saved feature importance plot to {output_path}")

    except Exception as e:
        logging.error(f"Error plotting feature importance: {str(e)}")
        plt.close()


def plot_tree_visualization(
    model: xgb.XGBClassifier,
    output_dir: Path,
    num_trees: Optional[int] = 3,
    figsize: Tuple[int, int] = (15, 10),
) -> None:
    """Create visualizations of individual trees from the XGBoost model.

    Args:
        model: Trained XGBoost classifier
        output_dir: Directory where tree plots should be saved
        num_trees: Number of trees to visualize
        figsize: Figure size (width, height)
    """
    try:
        if not hasattr(model, "n_estimators") or model.n_estimators is None:
            raise ValueError("Model does not have valid number of estimators")

        if num_trees is None or num_trees <= 0:
            raise ValueError("Invalid number of trees specified")

        actual_trees = min(num_trees, model.n_estimators)

        for i in range(actual_trees):
            try:
                output_file = output_dir / f"tree_{i}.png"

                plt.figure(figsize=figsize)
                xgb.plot_tree(model, num_trees=i)
                plt.title(f"Decision Tree {i+1}", size=14, pad=20, weight="bold")

                plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.5)
                plt.close()
                logging.info(f"Saved tree visualization {i+1}/{actual_trees}")

            except Exception as e:
                logging.warning(f"Failed to plot tree {i}: {str(e)}")
                plt.close()
                continue

    except Exception as e:
        logging.error(f"Error in tree visualization: {str(e)}")
        plt.close()
