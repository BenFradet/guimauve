import json
import os
from typing import Any

import torch
from safetensors.torch import save_file


def extract_dict(model: torch.nn.Module, dicts: list[str], output_dir: str) -> list[str]:
    """
    Extracts a dict from the model object

    Parameters:
    model (torch.nn.Module): loaded pytorch model
    dicts (list[str]): the dicts to extract from the model in dotted-path format
    output_dir (str): the directory in which to store the jsons

    Returns:
    the paths to the json files
    """
    res = []
    for dotted_path in dicts:
        file_path = os.path.join(output_dir, f"{dotted_path}.json")
        with open(file_path, "w") as f:
            data = _deep_getattr(model, dotted_path)
            json.dump({str(k): v for k, v in data.items()}, f)
        res.append(file_path)
    return res


def extract_embedding(model: torch.nn.Module, embeddings: list[str], output_dir: str) -> list[str]:
    """
    Extracts tensors from the model object as safetensors files

    Parameters:
    model (torch.nn.Module): loaded pytorch model
    embeddings (list[str]): the tensors to extract from the model in dotted-path format
    output_dir (str): the directory in which to store the safetensors files

    Returns:
    the paths to the safetensors files
    """
    res = []
    for dotted_path in embeddings:
        file_path = os.path.join(output_dir, f"{dotted_path}.safetensors")
        weights = {}
        module = _deep_getattr(model, dotted_path)
        if isinstance(module, torch.nn.ModuleDict):
            for name, child in module.items():
                weights[name] = child.weight.data
        elif isinstance(module, torch.nn.Module):
            weights[dotted_path.split(".")[-1]] = module.weight.data
        else:
            raise TypeError(f"'{dotted_path}' is not a nn.Module nor a nn.ModuleDict")
        save_file(weights, file_path)
        res.append(file_path)

    return res


def _deep_getattr(obj: Any, path: str) -> Any:
    for split in path.split("."):
        obj = getattr(obj, split)
    return obj
