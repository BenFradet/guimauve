use burn::prelude::Backend;

pub trait ModelPlugin<B: Backend> {
    type Request;
    type Response;
    type ModelInput;
    type ModelOutput;

    fn pre(&self, req: Self::Request, device: &B::Device) -> Self::ModelInput;
    fn post(&self, output: Self::ModelOutput) -> Self::Response;
}
