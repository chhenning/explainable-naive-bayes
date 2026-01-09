import json
from typing import List, Tuple, Dict


def create_dataset_from_json(file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Read.

    :param file_path: Path to json.
    :type file_path: str
    """

    with open(file_path) as f:
        data = json.load(f)

    return data["train"], data["test"]
