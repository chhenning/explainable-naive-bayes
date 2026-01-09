def accuracy(results) -> float:
    """
    Returns the percentage of texts correctly classified.

    :param results: List of tuples (label, predicted label)
    """
    return sum([r[0] == r[1] for r in results]) / len(results)
