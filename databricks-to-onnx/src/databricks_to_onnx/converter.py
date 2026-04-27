import mlflow
import onnx
import torch

from databricks_to_onnx.input_tensor_schema import InputTensorSchema


def load_model(model_location: str) -> torch.nn.Module:
    """
    Fetches a pytorch model from databricks
    c.f. https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/
    or locally

    Parameters:
    model_location (str): databricks catalog model uri, e.g.
    "models:/catalog.schema.model_name@champion",
    or path to a directory containing a .pth file, e.g. "model/"

    Returns:
    a torch.nn.Module ready for inference
    """
    mlflow.set_registry_uri("databricks-uc")
    model = mlflow.pytorch.load_model(model_location)

    # check whether the entrypoint is just a "wrapper model"
    if hasattr(model, 'model'):
        model = model.model
    # type narrowing so that the eval below doesn't give linting errors
    if not isinstance(model, torch.nn.Module):
      raise TypeError(f"Expected torch.nn.Module, got {type(model)}")

    # evaluation mode as opposed to training mode
    model.eval()
    return model


def convert_model(
    model: torch.nn.Module,
    input_tensor_schemas: list[str],
    output_path: str,
) -> None:
    """
    Converts a pytorch model to onnx format and writes it to the filesystem
    c.f.
    - https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
    - https://docs.pytorch.org/docs/stable/onnx_export.html#torch.onnx.export

    Parameters:
    model (torch.nn.Module): loaded pytorch model
    input_tensor_schemas (list[str]): the schemas of the input tensors to the model in the format
    'name:dtype:dim1,dim2'
    output_path (str): where to store the onnx file
    """
    inputs = {}

    for schema in input_tensor_schemas:
        tensor_schema = InputTensorSchema.parse(schema)
        inputs[tensor_schema.name] = torch.zeros(tensor_schema.shape, dtype=tensor_schema.dtype)

    # pytorch model is dynamic while onnx' is static
    # export runs the model with an input tensor to trace and capture every op
    # hence the need for a correctly shaped dummy input
    dummy_input = tuple(inputs.values())
    input_names = list(inputs.keys())

    torch.onnx.export(model, dummy_input, output_path, input_names=input_names)

    # check whether the export was successful
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
