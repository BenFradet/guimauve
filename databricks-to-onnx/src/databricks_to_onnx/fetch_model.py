import mlflow
import torch.nn as nn


def fetch_model(model_uri: str) -> nn.Module:
    """
    Loads a pytorch model from databricks
    c.f. https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/

    Parameters:
    model_uri (str): databricks catalog model path, e.g.
    "models:/catalog.schema.model_name@champion"

    Returns:
    a torch.nn.Module ready for inference
    """
    mlflow.set_registry_uri("databricks-uc")
    model = mlflow.pytorch.load_model(model_uri)

    # evaluation mode as opposed to training mode
    model.eval()
    return model
