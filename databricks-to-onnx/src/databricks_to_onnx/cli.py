import click

from databricks_to_onnx.converter import convert_model, fetch_model, load_model


@click.command()
@click.option(
    "-l",
    "--local",
    required=True,
    is_flag=True,
    help="Whether to fetch the model from databricks or a local file",
)
@click.option(
    "-m",
    "--model-path",
    required=True,
    help='''Unity Catalog model URI, e.g. "models:/catalog.schema.model_name@champion" or
    local path, e.g. "path/to/model.pth"''',
)
@click.option(
    "-i",
    "--input-tensor-schema",
    "input_tensor_schemas",
    required=True,
    multiple=True,
    help='Input tensor schema as "name:dtype:dim1,dim2". Repeatable.',
)
@click.option(
    "-o",
    "--output-path",
    required=True,
    help="Output path for the .onnx file.",
)
def cli(
    model_path: str,
    input_tensor_schemas: tuple[str, ...],
    output_path: str,
    local: bool,
) -> None:
    click.echo(f"Loading model from: {model_path}")
    pytorch_model = load_model(model_path) if local else fetch_model(model_path)

    click.echo("Exporting to ONNX")
    convert_model(pytorch_model, list(input_tensor_schemas), output_path)

    click.echo(f"Done, wrote output to {output_path}")
