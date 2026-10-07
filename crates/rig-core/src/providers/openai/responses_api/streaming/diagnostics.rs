//! Wire diagnostics for one Responses SSE stream.
//!
//! A stream that fails mid-body or ends without `response.completed` otherwise
//! surfaces only its transport cause. These counters let the terminal log name
//! the upstream response and say what the provider had sent before it stopped.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;

use serde::Deserialize;

/// Distinct event types tallied by name; later novel types share one bucket so
/// a misbehaving upstream cannot grow the map without bound.
const MAX_TRACKED_TYPES: usize = 32;
const OVERFLOW_TYPE: &str = "other";
const UNTYPED: &str = "untyped";

macro_rules! log_stream {
    ($level:expr, $diagnostics:expr, $provider:expr, $message:literal $(, $($extra:tt)+)?) => {
        tracing::event!(
            $level,
            provider = $provider,
            response_id = $diagnostics.response_id.as_deref(),
            events = $diagnostics.events,
            data_bytes = $diagnostics.data_bytes,
            last_event_type = $diagnostics.last_event_type.as_deref(),
            event_types = %EventTypes(&$diagnostics.event_types),
            undecodable = $diagnostics.undecodable,
            $($($extra)+,)?
            $message
        )
    };
}

#[derive(Debug, Default)]
pub(super) struct StreamDiagnostics {
    response_id: Option<String>,
    completed: bool,
    events: u64,
    data_bytes: u64,
    event_types: BTreeMap<String, u64>,
    last_event_type: Option<String>,
    undecodable: u64,
}

#[derive(Deserialize)]
struct EventType<'a> {
    #[serde(rename = "type", borrow)]
    kind: Option<Cow<'a, str>>,
}

impl StreamDiagnostics {
    /// Tally one non-empty SSE data payload by its JSON `type`.
    pub(super) fn observe(&mut self, data: &str) {
        self.events += 1;
        self.data_bytes += data.len() as u64;
        let kind = serde_json::from_str::<EventType<'_>>(data)
            .ok()
            .and_then(|event| event.kind)
            .unwrap_or(Cow::Borrowed(UNTYPED));
        let key = if self.event_types.contains_key(kind.as_ref())
            || self.event_types.len() < MAX_TRACKED_TYPES
        {
            kind.as_ref()
        } else {
            OVERFLOW_TYPE
        };
        *self.event_types.entry(key.to_owned()).or_default() += 1;
        self.last_event_type = Some(kind.into_owned());
    }

    /// Count a payload that did not decode as any known stream chunk.
    pub(super) fn undecodable(&mut self) {
        self.undecodable += 1;
    }

    /// Record the upstream response identity. Returns `true` the first time an
    /// id is seen so the caller can stamp it on the span before completion.
    pub(super) fn response(&mut self, id: &str, completed: bool) -> bool {
        self.completed |= completed;
        if self.response_id.is_some() {
            return false;
        }
        self.response_id = Some(id.to_owned());
        true
    }

    /// Log a stream that terminated with an error.
    pub(super) fn log_failure(&self, provider: &str, error: &dyn fmt::Display) {
        log_stream!(
            tracing::Level::ERROR,
            self,
            provider,
            "Responses stream failed",
            error = %error
        );
    }

    /// Log anomalies of a stream that ended without an error.
    pub(super) fn log_end(&self, provider: &str) {
        if !self.completed {
            log_stream!(
                tracing::Level::WARN,
                self,
                provider,
                "Responses stream ended without response.completed"
            );
        } else if self.undecodable > 0 {
            log_stream!(
                tracing::Level::WARN,
                self,
                provider,
                "Responses stream contained undecodable events"
            );
        }
    }
}

struct EventTypes<'a>(&'a BTreeMap<String, u64>);

impl fmt::Display for EventTypes<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, (kind, count)) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str(",")?;
            }
            write!(f, "{kind}={count}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use serde_json::json;

    use super::*;
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_json_events;
    use crate::providers::openai;
    use crate::test_utils::MockStreamingClient;
    use crate::test_utils::log_capture::CapturedLogs;

    #[test]
    fn tallies_event_types_and_bytes() {
        let mut diagnostics = StreamDiagnostics::default();
        diagnostics.observe(r#"{"type":"response.output_text.delta","delta":"a"}"#);
        diagnostics.observe(r#"{"type":"response.output_text.delta","delta":"b"}"#);
        diagnostics.observe(r#"{"delta":"no type"}"#);
        diagnostics.observe("not json");

        assert_eq!(diagnostics.events, 4);
        assert_eq!(
            EventTypes(&diagnostics.event_types).to_string(),
            "response.output_text.delta=2,untyped=2"
        );
        assert_eq!(diagnostics.last_event_type.as_deref(), Some(UNTYPED));
    }

    #[test]
    fn novel_types_beyond_the_cap_share_one_bucket() {
        let mut diagnostics = StreamDiagnostics::default();
        for index in 0..MAX_TRACKED_TYPES + 3 {
            diagnostics.observe(&format!(r#"{{"type":"kind.{index}"}}"#));
        }
        diagnostics.observe(r#"{"type":"kind.0"}"#);

        assert_eq!(diagnostics.event_types.len(), MAX_TRACKED_TYPES + 1);
        assert_eq!(diagnostics.event_types.get(OVERFLOW_TYPE), Some(&3));
        assert_eq!(diagnostics.event_types.get("kind.0"), Some(&2));
        assert_eq!(
            diagnostics.last_event_type.as_deref(),
            Some("kind.0"),
            "the last type keeps its real name even when tallied as overflow"
        );
    }

    /// Drain a mock Responses stream and return what it logged.
    async fn drain_logs(events: &[serde_json::Value]) -> String {
        let logs = CapturedLogs::at(tracing::Level::WARN);

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(events),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");
        while stream.next().await.is_some() {}

        logs.text()
    }

    fn created(id: &str) -> serde_json::Value {
        json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": id, "object": "response", "created_at": 0, "status": "in_progress",
                "model": "gpt-5.4", "output": [], "tools": [],
            },
        })
    }

    #[tokio::test]
    async fn stream_without_completion_logs_its_response_and_event_tally() {
        let logs = drain_logs(&[
            created("resp_stalled"),
            json!({
                "type": "response.output_text.delta", "output_index": 0,
                "content_index": 0, "sequence_number": 1, "delta": "hi",
            }),
            json!({"type": "response.web_search_call.searching", "output_index": 0}),
            json!({"type": "response.output_text.delta", "output_index": 0}),
        ])
        .await;

        assert!(
            logs.contains("Responses stream ended without response.completed"),
            "{logs}"
        );
        assert!(logs.contains("response_id=\"resp_stalled\""), "{logs}");
        assert!(logs.contains("events=4"), "{logs}");
        assert!(logs.contains("undecodable=1"), "{logs}");
        assert!(
            logs.contains(
                "event_types=response.created=1,response.output_text.delta=2,\
                 response.web_search_call.searching=1"
            ),
            "{logs}"
        );
    }

    #[tokio::test]
    async fn terminal_failure_logs_the_response_it_belongs_to() {
        let mut failed = created("resp_failed");
        failed["type"] = json!("response.failed");
        failed["response"]["status"] = json!("failed");
        failed["response"]["error"] = json!({"code": "server_error", "message": "boom"});

        let logs = drain_logs(&[created("resp_failed"), failed]).await;

        assert!(logs.contains("Responses stream failed"), "{logs}");
        assert!(logs.contains("response_id=\"resp_failed\""), "{logs}");
        assert!(logs.contains("server_error: boom"), "{logs}");
    }

    #[test]
    fn first_response_id_wins_and_completion_is_sticky() {
        let mut diagnostics = StreamDiagnostics::default();
        assert!(diagnostics.response("resp_1", false));
        assert!(!diagnostics.response("resp_2", true));

        assert_eq!(diagnostics.response_id.as_deref(), Some("resp_1"));
        assert!(diagnostics.completed);
    }
}
