//! Responses wire regressions: argument progress must not wait for item completion.

use std::{collections::HashMap, convert::Infallible, sync::Arc, time::Duration};

use axum::{Router, body::Body, response::Response, routing::post};
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc};

use super::{ItemChunkKind, StreamingCompletionChunk};
use crate::{
    client::CompletionClient,
    completion::CompletionModel,
    providers::openai,
    streaming::{StreamedAssistantContent, ToolCallDeltaContent},
};

fn argument_delta(index: u64, delta: &str) -> Value {
    json!({
        "type": "response.function_call_arguments.delta", "output_index": index,
        "item_id": format!("fc_{index}"), "sequence_number": 3 + index, "delta": delta,
    })
}

fn argument_done(index: u64, arguments: &str) -> Value {
    json!({
        "type": "response.function_call_arguments.done", "output_index": index,
        "item_id": format!("fc_{index}"), "sequence_number": 9 + index,
        "name": "write_file", "arguments": arguments,
    })
}

fn item(index: u64, done: bool, arguments: &str) -> Value {
    json!({
        "type": if done { "response.output_item.done" } else { "response.output_item.added" },
        "output_index": index, "sequence_number": if done { 11 + index } else { 1 + index },
        "item": {
            "type": "function_call", "id": format!("fc_{index}"),
            "call_id": format!("call_{index}"), "name": "write_file", "arguments": arguments,
            "status": if done { "completed" } else { "in_progress" },
        },
    })
}

#[test]
fn function_arguments_decode_without_content_index() {
    let chunk: StreamingCompletionChunk = serde_json::from_value(argument_delta(0, "{\"text\":"))
        .expect("real-shaped function argument delta must decode");
    assert!(matches!(chunk, StreamingCompletionChunk::Delta(chunk)
        if matches!(&chunk.data, ItemChunkKind::FunctionCallArgsDelta(delta)
            if delta.item_id == "fc_0" && delta.delta == "{\"text\":")));
    let chunk: StreamingCompletionChunk = serde_json::from_value(argument_done(0, "{}"))
        .expect("real-shaped function arguments done must decode");
    assert!(matches!(chunk, StreamingCompletionChunk::Delta(chunk)
        if matches!(chunk.data, ItemChunkKind::FunctionCallArgsDone(_))));
}

async fn send(sender: &mpsc::Sender<String>, event: Value) {
    sender
        .send(format!("data: {event}\n\n"))
        .await
        .expect("send SSE");
}

#[tokio::test]
async fn live_argument_deltas_arrive_before_completion_and_keep_call_identity() {
    tokio::time::timeout(Duration::from_secs(10), verify_live_argument_deltas())
        .await
        .expect("live tool lifecycle completes within test deadline");
}

async fn verify_live_argument_deltas() {
    let (sender, receiver) = mpsc::channel::<String>(16);
    let receiver = Arc::new(Mutex::new(Some(receiver)));
    let app = Router::new().route(
        "/responses",
        post(move || {
            let receiver = receiver.clone();
            async move {
                let receiver = receiver.lock().await.take().expect("one request");
                let body = futures::stream::unfold(receiver, |mut receiver| async move {
                    receiver
                        .recv()
                        .await
                        .map(|chunk| (Ok::<_, Infallible>(chunk), receiver))
                });
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(body))
                    .expect("SSE response")
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listen");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move { axum::serve(listener, app).await.expect("serve") });
    // Abort on assertion failure as well as on success.
    let _guard = ServerGuard(server);
    let client = openai::Client::builder()
        .api_key("test")
        .base_url(format!("http://{address}"))
        .build()
        .expect("client");
    let model = client.completion_model("gpt-5.4");
    let mut stream = model
        .stream(model.completion_request("write files").build())
        .await
        .expect("stream");
    send(
        &sender,
        json!({"type":"response.output_text.delta", "output_index":0,
        "content_index":0, "sequence_number":0, "delta":"Writing files."}),
    )
    .await;
    assert!(matches!(
        stream.next().await,
        Some(Ok(StreamedAssistantContent::Text(_)))
    ));
    let mut internal_ids = HashMap::new();
    for index in 0..2 {
        send(&sender, item(index, false, "")).await;
        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("name delivered while upstream open")
            .expect("name")
            .expect("valid name");
        let StreamedAssistantContent::ToolCallDelta {
            id,
            internal_call_id,
            content,
        } = event
        else {
            panic!("expected tool name, got {event:?}");
        };
        assert_eq!(id, format!("fc_{index}"));
        assert!(matches!(content, ToolCallDeltaContent::Name(name) if name == "write_file"));
        internal_ids.insert(index, internal_call_id);
    }
    assert_ne!(internal_ids[&0], internal_ids[&1]);
    let fragments = ["{\"text\":\"", "héllo ", "\\\"world\\\"\\n\"}"];
    for fragment in fragments {
        for index in 0..2 {
            send(&sender, argument_delta(index, fragment)).await;
            let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
                .await
                .expect("argument delta delivered before completion")
                .expect("delta")
                .expect("valid delta");
            let StreamedAssistantContent::ToolCallDelta {
                id,
                internal_call_id,
                content,
            } = event
            else {
                panic!("expected argument delta, got {event:?}");
            };
            assert_eq!(id, format!("fc_{index}"));
            assert_eq!(internal_call_id, internal_ids[&index]);
            assert!(matches!(content, ToolCallDeltaContent::Delta(args) if args == fragment));
        }
    }
    let arguments = fragments.concat();
    for index in 0..2 {
        send(&sender, argument_done(index, &arguments)).await;
        send(&sender, item(index, true, &arguments)).await;
    }
    send(
        &sender,
        json!({"type":"response.completed", "sequence_number":13,
        "response":{"id":"resp_test", "object":"response", "created_at":0,
            "status":"completed", "model":"gpt-5.4", "output":[],
            "usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30,
                "output_tokens_details":{"reasoning_tokens":0}}}}),
    )
    .await;
    drop(sender);
    for index in 0..2 {
        let event = stream
            .next()
            .await
            .expect("completed call")
            .expect("valid call");
        let StreamedAssistantContent::ToolCall {
            tool_call,
            internal_call_id,
        } = event
        else {
            panic!("expected completed call, got {event:?}");
        };
        assert_eq!(internal_call_id, internal_ids[&index]);
        assert_eq!(tool_call.id, format!("fc_{index}"));
        assert_eq!(tool_call.call_id, Some(format!("call_{index}")));
        assert_eq!(
            tool_call.function.arguments,
            json!({"text":"héllo \"world\"\n"})
        );
    }
    assert!(
        matches!(stream.next().await, Some(Ok(StreamedAssistantContent::Final(response)))
        if response.usage.total_tokens == 30)
    );
    assert!(stream.next().await.is_none());
}

struct ServerGuard(tokio::task::JoinHandle<()>);

impl Drop for ServerGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}
