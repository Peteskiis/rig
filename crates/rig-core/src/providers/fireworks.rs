//! Fireworks AI API client and Rig integration.

use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client::{self, HttpClientExt},
};

/// Fireworks' OpenAI-compatible Chat Completions base URL.
pub const FIREWORKS_API_BASE_URL: &str = "https://api.fireworks.ai/inference/v1";

/// Fireworks-hosted Kimi K3 model identifier.
pub const KIMI_K3: &str = "accounts/fireworks/models/kimi-k3";

/// Fireworks provider extension used by Rig's generic client.
#[derive(Debug, Default, Clone, Copy)]
pub struct FireworksExt;

/// Builder state for the Fireworks provider extension.
#[derive(Debug, Default, Clone, Copy)]
pub struct FireworksBuilder;

type FireworksApiKey = BearerAuth;

/// Fireworks client using Rig's generic HTTP backend.
pub type Client<H = reqwest::Client> = client::Client<FireworksExt, H>;
/// Builder for a Fireworks client.
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<FireworksBuilder, FireworksApiKey, H>;

impl Provider for FireworksExt {
    type Builder = FireworksBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl<H> Capabilities<H> for FireworksExt {
    type Completion = Capable<super::openai::completion::GenericCompletionModel<FireworksExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for FireworksExt {}

impl super::openai::completion::OpenAICompatibleProvider for FireworksExt {
    const PROVIDER_NAME: &'static str = "fireworks";
}

impl ProviderBuilder for FireworksBuilder {
    type Extension<H>
        = FireworksExt
    where
        H: HttpClientExt;
    type ApiKey = FireworksApiKey;

    const BASE_URL: &'static str = FIREWORKS_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(FireworksExt)
    }
}

impl ProviderClient for Client {
    type Input = FireworksApiKey;
    type Error = crate::client::ProviderClientError;

    /// Creates a Fireworks client from `FIREWORKS_API_KEY`.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("FIREWORKS_API_KEY")?;
        let mut builder = Self::builder().api_key(api_key);

        if let Some(base_url) = crate::client::optional_env_var("FIREWORKS_API_BASE")? {
            builder = builder.base_url(base_url);
        }

        builder.build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(input).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{Arc, Mutex},
    };

    use tracing::{Subscriber, dispatcher::Dispatch, field::Visit};
    use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

    use super::*;
    use crate::{
        OneOrMany,
        client::{CompletionClient, VerifyClient},
        completion::{CompletionModel, CompletionRequest},
        message::{AssistantContent, Message},
        test_utils::{MockStreamingClient, RecordingHttpClient},
    };

    #[derive(Clone, Default)]
    struct CapturedSpanFields(Arc<Mutex<Vec<BTreeMap<String, String>>>>);

    impl CapturedSpanFields {
        fn provider_names(&self) -> Vec<String> {
            self.0
                .lock()
                .expect("captured span fields lock")
                .iter()
                .filter_map(|fields| fields.get("gen_ai.provider.name").cloned())
                .collect()
        }
    }

    #[derive(Default)]
    struct FieldVisitor(BTreeMap<String, String>);

    impl Visit for FieldVisitor {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }

        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
    }

    impl<S> Layer<S> for CapturedSpanFields
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(
            &self,
            attributes: &tracing::span::Attributes<'_>,
            _: &tracing::Id,
            _: Context<'_, S>,
        ) {
            let mut visitor = FieldVisitor::default();
            attributes.record(&mut visitor);
            self.0
                .lock()
                .expect("captured span fields lock")
                .push(visitor.0);
        }
    }

    fn replay_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::many(vec![
                Message::user("Use the calculator."),
                Message::Assistant {
                    id: None,
                    content: OneOrMany::many(vec![
                        AssistantContent::reasoning("2 + 2 requires a tool call"),
                        AssistantContent::text("I will calculate that."),
                        AssistantContent::tool_call(
                            "call_1",
                            "calculator",
                            serde_json::json!({"expression": "2 + 2"}),
                        ),
                    ])
                    .expect("assistant replay content is non-empty"),
                },
                Message::tool_result("call_1", "4"),
            ])
            .expect("chat history is non-empty"),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        }
    }

    #[tokio::test]
    async fn verifies_and_replays_reasoning_text_and_tool_calls() {
        let http = RecordingHttpClient::new(
            r#"{"id":"completion_1","object":"chat.completion","created":0,"model":"accounts/fireworks/models/kimi-k3","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"total_tokens":4}}"#,
        );
        let client = Client::builder()
            .api_key("test-key")
            .base_url("https://fireworks.test/inference/v1")
            .http_client(http.clone())
            .build()
            .expect("build Fireworks client");

        client.verify().await.expect("verify through /models");
        client
            .completion_model(KIMI_K3)
            .completion(replay_request())
            .await
            .expect("complete replay request");

        let requests = http.requests();
        assert_eq!(
            requests[0].uri,
            "https://fireworks.test/inference/v1/models"
        );
        assert_eq!(
            requests[1].uri,
            "https://fireworks.test/inference/v1/chat/completions"
        );

        let body: serde_json::Value =
            serde_json::from_slice(&requests[1].body).expect("Fireworks request JSON");
        let assistant = &body["messages"][1];
        assert_eq!(assistant["reasoning_content"], "2 + 2 requires a tool call");
        assert_eq!(assistant["content"][0]["text"], "I will calculate that.");
        assert_eq!(assistant["tool_calls"][0]["id"], "call_1");
    }

    #[test]
    fn unary_and_streaming_spans_identify_fireworks() {
        let captured = CapturedSpanFields::default();
        let dispatch = Dispatch::new(tracing_subscriber::registry().with(captured.clone()));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime");

        tracing::dispatcher::with_default(&dispatch, || {
            runtime.block_on(async {
                let unary = Client::builder()
                    .api_key("test-key")
                    .http_client(RecordingHttpClient::new(
                        r#"{"id":"completion_1","object":"chat.completion","created":0,"model":"accounts/fireworks/models/kimi-k3","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"total_tokens":4}}"#,
                    ))
                    .build()
                    .expect("build unary client");
                unary
                    .completion_model(KIMI_K3)
                    .completion(replay_request())
                    .await
                    .expect("unary completion");

                let streaming = Client::builder()
                    .api_key("test-key")
                    .http_client(MockStreamingClient::default())
                    .build()
                    .expect("build streaming client");
                streaming
                    .completion_model(KIMI_K3)
                    .stream(replay_request())
                    .await
                    .expect("stream setup");
            });
        });

        assert_eq!(captured.provider_names(), vec!["fireworks", "fireworks"]);
    }

    #[test]
    fn public_constants_describe_the_fireworks_chat_completions_api() {
        assert_eq!(
            FIREWORKS_API_BASE_URL,
            "https://api.fireworks.ai/inference/v1"
        );
        assert_eq!(KIMI_K3, "accounts/fireworks/models/kimi-k3");
    }
}
