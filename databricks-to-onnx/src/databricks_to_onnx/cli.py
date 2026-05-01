import click

from databricks_to_onnx.converter import convert_model, load_model
from databricks_to_onnx.extractor import extract_dict


@click.command()
@click.option(
    "-m",
    "--model-location",
    required=True,
    help='''Unity Catalog model URI,
    e.g. "models:/catalog.schema.model_name@champion",
    or local directory, e.g. "path/to/champion/"''',
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
    "-e",
    "--extract",
    "extracts",
    required=False,
    multiple=True,
    help='''Extract a dict you want to get out of the model in dotted path format, e.g.
    "maps.vocab". Will be written out as "maps.vocab.json" in the output directory. Repeatable.''',
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    help="Output dir for the model.onnx file.",
)
def cli(
    model_location: str,
    input_tensor_schemas: tuple[str, ...],
    extracts: tuple[str, ...],
    output_dir: str,
) -> None:
    click.echo(f"Loading model from: {model_location}")
    pytorch_model = load_model(model_location)

    extract_list = list(extracts)
    if extract_list:
        click.echo("Extracting:")
        for extracted in extract_dict(pytorch_model, extract_list, output_dir):
            click.echo(f"- {extracted}")

    click.echo("Exporting to ONNX")
    convert_model(pytorch_model, list(input_tensor_schemas), output_dir)

    click.echo(f"Done, wrote output to {output_dir}/model.onnx")
