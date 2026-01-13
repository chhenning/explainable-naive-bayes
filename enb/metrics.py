from typing import List, Tuple, Dict
from enb.printing_helpers import _print_table


def accuracy(results: List[Tuple[str, str]]) -> float:
    """
    Returns the percentage of texts correctly classified.

    :param results: List of tuples (true_label, predicted_label)
    """
    if not results:
        return 0.0
    return sum([r[0] == r[1] for r in results]) / len(results)


def confusion_matrix(results: List[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    """
    Computes a confusion matrix: cm[true_label][predicted_label] = count.
    """
    cm = {}
    # Get all unique labels to ensure a square matrix
    labels = sorted(list(set([r[0] for r in results] + [r[1] for r in results])))

    for true_label in labels:
        cm[true_label] = {pred_label: 0 for pred_label in labels}

    for true_label, pred_label in results:
        cm[true_label][pred_label] += 1

    return cm


def print_confusion_matrix(cm: Dict[str, Dict[str, int]]) -> None:
    """
    Pretty-prints the confusion matrix.
    """
    labels = sorted(cm.keys())
    headers = ["true \\ pred"] + labels
    rows = []
    for true_label in labels:
        row = [true_label]
        for pred_label in labels:
            row.append(cm[true_label].get(pred_label, 0))
        rows.append(row)

    _print_table(headers, rows)

