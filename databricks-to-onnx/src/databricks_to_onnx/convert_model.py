import onnx
import torch

from databricks_to_onnx.input_tensor_schema import InputTensorSchema


def convert_model(
    model: torch.nn.Module,
    input_tensor_schemas: list[str],
    output_path: str,
) -> None:
    inputs = {}
    input_names = []

    for schema in input_tensor_schemas:
        schema = InputTensorSchema.parse(schema)
        inputs[schema.name] = torch.zeros(schema.shape, dtype=schema.dtype)
        input_names.append(schema.name)

    dummy_input = tuple(inputs.values())

    torch.onnx.export(model, dummy_input, output_path, input_names=input_names)

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
