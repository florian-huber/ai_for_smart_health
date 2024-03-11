import numpy as np
import matplotlib.pyplot as plt
from ai_smart_health.model_evaluation import plot_roc_curves

import matplotlib

matplotlib.use("Agg")


# Define a mockup of labels, predictions, and class names for testing
labels = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1]])

predictions = np.array([[0.1, 0.9, 0.8], [0.8, 0.2, 0.7], [0.6, 0.7, 0.3], [0.3, 0.4, 0.9]])

classes = ["Class1", "Class2", "Class3"]


def test_plot_roc_curves():
    # fig, ax = plt.subplots()  # Create a figure and axes for plotting
    ax = plt.gca()
    try:
        plot_roc_curves(labels, predictions, classes, ax=ax)
    except Exception as e:
        assert False, f"Function raised exception: {e}"

    # Test if the correct number of ROC curves is plotted
    assert len(ax.lines) == labels.shape[1] + 1, "Incorrect number of ROC curves plotted"

    # Test if the axes labels are set correctly
    assert ax.get_xlabel() == "False Positive Rate", "Incorrect x-axis label"
    assert ax.get_ylabel() == "True Positive Rate", "Incorrect y-axis label"
    assert ax.get_title() == "Receiver Operating Characteristic for Multi-class Prediction", "Incorrect title"

    # Test if the legend content is correct
    # legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    # expected_legend_texts = [f"{classes[i]} (AUC = {1.0:.2f})" for i in range(len(classes))]
    # assert legend_texts == expected_legend_texts, "Incorrect legend content"

    # Test if the plot appearance is as expected
    assert ax.get_xlim() == (0.0, 1.0), "Incorrect x-axis limits"
    assert ax.get_ylim() == (0.0, 1.0), "Incorrect y-axis limits"

    # Test if grid is enabled
    assert ax.get_xgridlines() != [] and ax.get_ygridlines() != [], "Grid is not enabled"

    # Test if the random classifier line is plotted
    assert len(ax.lines) > labels.shape[1], "Random classifier line is not plotted"

    plt.close()  # Close the plot after testing
