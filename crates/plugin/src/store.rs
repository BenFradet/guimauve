use anyhow::{Context, Error, Result};
use burn::{Tensor, prelude::Backend, tensor::BasicOps};
use burn_store::{ModuleStore, SafetensorsStore};

// TODO: might make sense to upstream this
pub fn get_tensor<'a, B: Backend, const D: usize, K: BasicOps<B>>(
    store: &'a mut SafetensorsStore,
    device: &B::Device,
    tensor_name: &'a str,
) -> Result<Tensor<B, D, K>> {
    store
        .get_snapshot(tensor_name)
        .map_err(Error::msg)?
        .context(format!(
            "missing {tensor_name} tensor from safetensors file"
        ))
        .and_then(|snap| snap.to_data().map_err(Error::msg))
        .map(|td| Tensor::<B, D, K>::from_data(td, device))
}
