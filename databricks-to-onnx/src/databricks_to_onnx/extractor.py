import json
import os
from typing import Any

import torch


def extract_dict(model: torch.nn.Module, extracts: list[str], output_dir: str) -> list[str]:
    """
    Extracts a dict from the model object

    Parameters:
    model (torch.nn.Module): loaded pytorch model
    extracts (list[str]): the dicts to extract from the model in dotted-path format
    output_dir (str): the directory in which to store the jsons

    Returns:
    the paths to the json files
    """
    res = []
    for dotted_path in extracts:
        file_path = os.path.join(output_dir, f"{dotted_path}.json")
        with open(file_path, "w") as f:
            data = deep_getattr(model, dotted_path)
            json.dump({str(k): v for k, v in data.items()}, f)
        res.append(file_path)
    return res


def deep_getattr(obj: Any, path: str) -> Any:
    for split in path.split("."):
        obj = getattr(obj, split)
    return obj
