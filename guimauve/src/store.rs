use anyhow::{Context, Result};
use burn::{Tensor, prelude::Backend, tensor::BasicOps};
use burn_store::{ModuleStore, SafetensorsStore};

// TODO: might make sense to upstream this
/// Gets a tensor from a safetensors file using its name.
///
/// # Arguments
///
/// * `store` - The safetensors store
/// * `device` - The burn device being used
/// * `tensor_name` - The tensor name/path (e.g., "encoder.layer1.weight")
///
/// # Errors
///
/// Returns a [`anyhow::Error`] if:
/// - an error occurred accessing storage
/// - the tensor name is missing from the store
/// - the tensor data fails to load
pub fn get_tensor<B: Backend, const D: usize, K: BasicOps<B>>(
    store: &mut SafetensorsStore,
    device: &B::Device,
    tensor_name: &str,
) -> Result<Tensor<B, D, K>> {
    let snapshot = store
        .get_snapshot(tensor_name)?
        .with_context(|| format!("missing {tensor_name} tensor from safetensors file"))?;
    let data = snapshot.to_data()?;
    Ok(Tensor::<B, D, K>::from_data(data, device))
}
