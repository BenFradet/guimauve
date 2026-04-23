import torch

DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}


def parse_tensor_spec(spec: str) -> tuple[str, torch.dtype, list[int]]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected 'name:dtype:shape', got '{spec}'")

    name = parts[0]

    dtype = DTYPES.get(parts[1])
    if dtype is None:
        raise ValueError(f"Unknown type '{parts[1]}', available: '{list(DTYPES)}'")

    shape = [int(d) for d in parts[2].split(",")]
    return name, dtype, shape
