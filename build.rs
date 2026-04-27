use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("models/model.onnx")
        .out_dir("models/")
        .run_from_script();
}
