//! Provider integrations included in `rig-core`.
//!
//! - Anthropic
//! - Azure OpenAI
//! - ChatGPT and GitHub Copilot auth-backed clients
//! - Cohere
//! - DeepSeek
//! - Fireworks AI
//! - Galadriel
//! - Gemini
//! - Groq
//! - Hugging Face
//! - Hyperbolic
//! - Llamafile
//! - MiniMax
//! - Mira
//! - Mistral
//! - Moonshot
//! - Ollama
//! - OpenAI
//! - OpenRouter
//! - Perplexity
//! - Together
//! - Voyage AI
//! - xAI
//! - Xiaomi MiMo
//! - Z.ai
//!
//! Each provider module defines a `Client` type and model types for the
//! capabilities it supports. Capability traits such as
//! [`CompletionClient`](crate::client::CompletionClient) and
//! [`EmbeddingsClient`](crate::client::EmbeddingsClient) are implemented only
//! when the provider declares that capability.
//!
//! # Example
//! ```no_run
//! use rig_core::{
//!     agent::AgentBuilder,
//!     client::{CompletionClient, ProviderClient},
//!     providers::openai,
//! };
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize the OpenAI client
//! let openai = openai::Client::from_env()?;
//!
//! // Create a model and initialize an agent
//! let model = openai.completion_model(openai::GPT_5_2);
//!
//! let agent = AgentBuilder::new(model)
//!     .preamble("\
//!         You are Gandalf the white and you will be conversing with other \
//!         powerful beings to discuss the fate of Middle Earth.\
//!     ")
//!     .build();
//!
//! // Alternatively, you can initialize an agent directly
//! let agent = openai.agent(openai::GPT_5_2)
//!     .preamble("\
//!         You are Gandalf the white and you will be conversing with other \
//!         powerful beings to discuss the fate of Middle Earth.\
//!     ")
//!     .build();
//! # Ok(())
//! # }
//! ```
pub mod anthropic;
pub mod azure;
pub mod chatgpt;
pub mod cohere;
pub mod copilot;
pub mod deepseek;
pub mod fireworks;
pub mod galadriel;
pub mod gemini;
pub mod groq;
pub mod huggingface;
pub mod hyperbolic;
pub(crate) mod internal;
pub mod llamafile;
pub mod minimax;
pub mod mira;
pub mod mistral;
pub mod moonshot;
pub mod ollama;
pub mod openai;
pub mod openrouter;
pub mod perplexity;
pub mod together;
pub mod voyageai;
pub mod xai;
pub mod xiaomimimo;
pub mod zai;

pub(crate) fn json_with_redacted_images(
    value: &impl serde::Serialize,
) -> Result<String, serde_json::Error> {
    let mut value = serde_json::to_value(value)?;
    redact_image_payloads(&mut value, false);
    serde_json::to_string_pretty(&value)
}

fn redact_image_payloads(value: &mut serde_json::Value, image_context: bool) {
    match value {
        serde_json::Value::Object(fields) => {
            let image_context = image_context
                || fields
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|kind| matches!(kind, "image" | "image_url" | "input_image"));
            for (name, child) in fields {
                let nested_image_object = name == "image_url" && child.is_object();
                if image_context
                    && !nested_image_object
                    && matches!(name.as_str(), "data" | "url" | "image_url")
                {
                    *child = serde_json::Value::String("<redacted-image>".into());
                } else {
                    redact_image_payloads(child, image_context);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                redact_image_payloads(item, image_context);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod redaction_tests {
    use serde_json::json;

    #[test]
    fn image_payloads_are_redacted_without_hiding_safe_metadata() {
        let request = json!({
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": "SECRET_1"
                }},
                {"type": "image_url", "image_url": {"url": "SECRET_2", "detail": "high"}},
                {"type": "input_image", "image_url": "SECRET_3"},
                {"type": "text", "text": "keep me"}
            ]
        });

        let output = super::json_with_redacted_images(&request).unwrap();

        assert!(!output.contains("SECRET"));
        assert!(output.contains("<redacted-image>"));
        assert!(output.contains("image/png"));
        assert!(output.contains("high"));
        assert!(output.contains("keep me"));
    }
}
