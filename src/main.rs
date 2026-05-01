pub mod model {
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}

use burn::backend::{Flex, flex::FlexDevice};
use burn::tensor::{Int, Tensor};
use model::Model;

fn main() {
    let device = FlexDevice;

    let model: Model<Flex> = Model::default();

    let ss = Tensor::<Flex, 3, Int>::zeros([1, 5, 400], &device);
    let ass = Tensor::<Flex, 2, Int>::zeros([1, 400], &device);
    let gs = Tensor::<Flex, 2, Int>::zeros([1, 400], &device);
    let sts = Tensor::<Flex, 3, Int>::zeros([1, 5, 400], &device);
    let sas = Tensor::<Flex, 2, Int>::zeros([1, 400], &device);
    let sgs = Tensor::<Flex, 2, Int>::zeros([1, 400], &device);
    let fts = Tensor::<Flex, 2, Int>::zeros([1, 400], &device);
    let cts = Tensor::<Flex, 2, Int>::zeros([1, 400], &device);

    let output = model.forward(ss, ass, gs, sts, sas, sgs, fts, cts);
    println!("output: {output:?}");
}
