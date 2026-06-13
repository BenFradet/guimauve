import os

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
    "-d",
    "--dict",
    "dicts",
    required=False,
    multiple=True,
    help="""Extract a dict you want to get out of the model in dotted path format, e.g.
    "maps.vocab". Will be written out as "maps.vocab.json" in the output directory. Repeatable.""",
)
@click.option(
    "-e",
    "--embedding",
    "embeddings",
    required=False,
    multiple=True,
    help="""Extract embeddings from a nn.ModuleDict you want to get out of the model, as a
    safetensors file, in dotted path format, e.g. "embeddings.input". Will be written out as
    "embeddings.input.safetensors" in the output directory. Repeatable.""",
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
    dicts: tuple[str, ...],
    embeddings: tuple[str, ...],
    output_dir: str,
) -> None:
    click.echo(f"Loading model from: {model_location}")
    pytorch_model = load_model(model_location)

    dict_list = list(dicts)
    if dict_list:
        click.echo("Extracting dicts:")
        for extracted in extract_dict(pytorch_model, dict_list, output_dir):
            click.echo(f"- {extracted}")

    embedding_list = list(embeddings)
    if embedding_list:
        click.echo("Extracting embeddings:")

    click.echo("Exporting to ONNX")
    convert_model(pytorch_model, list(input_tensor_schemas), output_dir)

    output_path = os.path.join(output_dir, "model.onnx")
    click.echo(f"Done, wrote output to {output_path}")
