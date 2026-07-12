from dataclasses import dataclass

import torch

DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}


@dataclass
class InputTensorSchema:
    name: str
    dtype: torch.dtype
    shape: list[int]

    @staticmethod
    def parse(schema: str) -> "InputTensorSchema":
        parts = schema.split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected 'name:dtype:shape', got '{schema}'")

        name = parts[0]

        dtype = DTYPES.get(parts[1])
        if dtype is None:
            raise ValueError(f"Unknown type '{parts[1]}', available: '{list(DTYPES)}'")

        shape = [int(d) for d in parts[2].split(",")]
        return InputTensorSchema(name, dtype, shape)
