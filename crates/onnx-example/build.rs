use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("../../models/pt_to_en/transformer.onnx")
        .out_dir("model/")
        .run_from_script();
}
