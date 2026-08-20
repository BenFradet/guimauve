use serde::{Serialize, de::DeserializeOwned};

/// Trait which drives the interaction between the [`Server`] and the model.
///
/// [`Server`]: crate::server::Server
pub trait ModelPlugin: Send + Sync + 'static {
    /// Type of the HTTP request, needs to be [`DeserializeOwned`].
    /// c.f. <https://serde.rs/lifetimes.html#trait-bounds>
    type Request: DeserializeOwned + Send + 'static;

    /// Type of the HTTP response, needs to be [`Serialize`].
    type Response: Serialize + Send + 'static;

    /// Type representing the model's input.
    type ModelInput;

    /// Type representing the model's output.
    type ModelOutput;

    /// Type representing possible errors during parsing or inference.
    type Error: std::fmt::Display + Send;

    /// Processes the incoming HTTP request and turns it into an input the model understands.
    ///
    /// # Errors
    ///
    /// Can return an [`Self::Error`] if parsing the HTTP request fails.
    fn pre(&self, req: Self::Request) -> Result<Self::ModelInput, Self::Error>;

    /// Calls the model to get an output, typically involves calling `forward` on the model.
    ///
    /// # Errors
    ///
    /// Can return an [`Self::Error`] if the inference call fails.
    fn infer(&self, input: Self::ModelInput) -> Result<Self::ModelOutput, Self::Error>;

    /// Turns the model's output into an HTTP response.
    ///
    /// # Errors
    ///
    /// Can return an [`Self::Error`] if post processing fails.
    fn post(&self, output: Self::ModelOutput) -> Result<Self::Response, Self::Error>;
}
