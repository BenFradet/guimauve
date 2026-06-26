use burn_onnx::{LoadStrategy, ModelGen};

fn main() {
    ModelGen::new()
        .input("../../models/pt_to_en/transformer.onnx")
        .out_dir("model/")
        .load_strategy(LoadStrategy::Embedded)
        .run_from_script();
}
