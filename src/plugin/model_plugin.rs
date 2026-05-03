pub trait ModelPlugin {
    type Request;
    type Response;
    type ModelInput;
    type ModelOutput;
    type Error;

    fn pre(&self, req: Self::Request) -> Result<Self::ModelInput, Self::Error>;
    fn infer(&self, input: Self::ModelInput) -> Result<Self::ModelOutput, Self::Error>;
    fn post(&self, output: Self::ModelOutput) -> Result<Self::Response, Self::Error>;
}
