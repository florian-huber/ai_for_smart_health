"""
Small helper functions for evaluating machine learning models.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curves(
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: list,
    ax=None,
    sorted: bool = True,
    colormap: str = "viridis_r",
):
    """Plot ROC curves for multi-class prediction.

    Computes ROC curve and ROC area for each class and plots them.

    Args:
        labels (numpy.ndarray): Ground truth labels, shape (n_samples, n_classes).
        predictions (numpy.ndarray): Predicted probabilities, shape (n_samples, n_classes).
        classes (list): List of class names.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If not provided, a new plot will be created.
        sorted (bool, optional): Whether to sort classes based on AUC score (default True).
        colormap (str, optional): Matplotlib colormap for color mapping (default "viridis_r").

    Example:

    .. code-block:: python

        plot_roc_curves(labels, predictions, classes)
        plt.show()

    """
    # Initialize dictionaries for false positive rate, true positive rate, and list for AUC scores
    fpr = {}
    tpr = {}
    roc_auc = []

    # Validate input shapes
    if not labels.shape[1] == predictions.shape[1] == len(classes):
        raise ValueError("Number of classes in labels and predictions is not the same.")
    if not labels.shape[0] == predictions.shape[0]:
        raise ValueError("Number of labels and predictions is not the same.")

    n_classes = labels.shape[1]

    # Compute ROC curve and AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
        roc_auc.append(auc(fpr[i], tpr[i]))

    # Sort classes by AUC if specified
    if sorted:
        indices = np.argsort(np.array(roc_auc))[::-1]
    else:
        indices = np.arange(n_classes)

    # Set up color mapping
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, n_classes))

    # Plot all ROC curves
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    for count, i in enumerate(indices):
        ax.plot(fpr[i], tpr[i], color=colors[count], lw=2, label=f"{classes[i]} (AUC = {roc_auc[i]:0.2f})")

    # Plot the random classifier line
    ax.plot([0, 1], [0, 1], "k--")

    # Set plot limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic for Multi-class Prediction")

    # Add legend and grid
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True)
