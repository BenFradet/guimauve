import click

from databricks_to_onnx.converter import convert_model, fetch_model


@click.command()
@click.option(
    "-m", "--model-uri",
    required=True,
    help='Unity Catalog model URI, e.g. "models:/catalog.schema.model_name@champion".',
)
@click.option(
    "-i", "--input-tensor-schema",
    "input_tensor_schemas",
    required=True,
    multiple=True,
    help='Input tensor schema as "name:dtype:dim1,dim2". Repeatable.',
)
@click.option(
    "-o", "--output-path",
    required=True,
    help="Output path for the .onnx file.",
)
def cli(model_uri: str, input_tensor_schemas: tuple[str, ...], output_path: str) -> None:
    click.echo(f"Loading model from: {model_uri}")
    pytorch_model = fetch_model(model_uri)

    click.echo("Exporting to ONNX")
    convert_model(pytorch_model, list(input_tensor_schemas), output_path)

    click.echo(f"Done, wrote output to {output_path}")
