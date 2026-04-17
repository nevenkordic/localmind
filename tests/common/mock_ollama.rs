//! Tiny scripted Ollama mock for end-to-end tests.
//!
//! Listens on 127.0.0.1:<random port>, parses HTTP/1.1 requests by hand
//! (no extra deps beyond tokio, which is already in the main crate), and
//! returns scripted replies routed by request path:
//!
//!   POST /api/chat   -> next chat reply from the queue (FIFO)
//!   POST /api/embed  -> deterministic 768-dim zero vector
//!   GET  /api/tags   -> empty model list
//!
//! All received requests are recorded so tests can assert on what the agent
//! sent. The mock auto-shuts-down when the `MockOllama` value is dropped.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

#[derive(Clone, Debug)]
pub struct MockReply {
    pub status: u16,
    pub body: String,
}

impl MockReply {
    pub fn ok_json(body: serde_json::Value) -> Self {
        Self {
            status: 200,
            body: body.to_string(),
        }
    }

    /// Build a `/api/chat` reply from a plain assistant message (no tool calls).
    pub fn chat_text(content: &str) -> Self {
        Self::ok_json(serde_json::json!({
            "message": {
                "role": "assistant",
                "content": content,
            },
            "done": true
        }))
    }

    /// Build a `/api/chat` reply that asks for a single structured tool call.
    pub fn chat_tool_call(name: &str, arguments: serde_json::Value) -> Self {
        Self::ok_json(serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                }]
            },
            "done": true
        }))
    }

    /// Build a `/api/chat` reply where the model emits a text-form tool call
    /// in `content` instead of structured `tool_calls` (qwen's quirk).
    pub fn chat_text_form_tool_call(text: &str) -> Self {
        Self::ok_json(serde_json::json!({
            "message": {
                "role": "assistant",
                "content": text,
            },
            "done": true
        }))
    }
}

#[derive(Debug, Clone)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub body: String,
}

pub struct MockOllama {
    pub url: String,
    pub requests: Arc<Mutex<Vec<RecordedRequest>>>,
    chat_replies: Arc<Mutex<VecDeque<MockReply>>>,
    _shutdown: oneshot::Sender<()>,
}

impl MockOllama {
    pub async fn start(chat_replies: Vec<MockReply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
        let addr = listener.local_addr().expect("addr");
        let url = format!("http://{}", addr);

        let chat_q: Arc<Mutex<VecDeque<MockReply>>> = Arc::new(Mutex::new(chat_replies.into()));
        let requests: Arc<Mutex<Vec<RecordedRequest>>> = Arc::new(Mutex::new(Vec::new()));
        let (tx, mut rx) = oneshot::channel::<()>();

        let chat_q_c = chat_q.clone();
        let requests_c = requests.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut rx => break,
                    accept = listener.accept() => {
                        if let Ok((stream, _)) = accept {
                            let chat_q = chat_q_c.clone();
                            let reqs = requests_c.clone();
                            tokio::spawn(async move {
                                let _ = handle_request(stream, chat_q, reqs).await;
                            });
                        }
                    }
                }
            }
        });

        Self {
            url,
            requests,
            chat_replies: chat_q,
            _shutdown: tx,
        }
    }

    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }

    pub fn remaining_chat_replies(&self) -> usize {
        self.chat_replies.lock().unwrap().len()
    }
}

async fn handle_request(
    mut stream: TcpStream,
    chat_q: Arc<Mutex<VecDeque<MockReply>>>,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
) -> std::io::Result<()> {
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0usize;

    // Read headers (until \r\n\r\n).
    let header_end = loop {
        let n = stream.read(&mut buf[total..]).await?;
        if n == 0 {
            return Ok(());
        }
        total += n;
        if let Some(end) = find_subseq(&buf[..total], b"\r\n\r\n") {
            break end;
        }
        if total == buf.len() {
            return Ok(()); // headers oversized, give up
        }
    };

    let header_text = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let content_length = header_text
        .lines()
        .find_map(|l| {
            let lower = l.to_ascii_lowercase();
            lower
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
        })
        .unwrap_or(0);

    let body_start = header_end + 4;
    while total < body_start + content_length {
        let n = stream.read(&mut buf[total..]).await?;
        if n == 0 {
            break;
        }
        total += n;
    }

    let mut request_line = header_text.lines();
    let first = request_line.next().unwrap_or("");
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let body = String::from_utf8_lossy(&buf[body_start..(body_start + content_length).min(total)])
        .into_owned();

    requests.lock().unwrap().push(RecordedRequest {
        method: method.clone(),
        path: path.clone(),
        body,
    });

    let reply = route(&path, &chat_q);

    let resp = format!(
        "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        reply.status,
        reply.body.len(),
        reply.body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

fn route(path: &str, chat_q: &Arc<Mutex<VecDeque<MockReply>>>) -> MockReply {
    if path.starts_with("/api/chat") {
        return chat_q.lock().unwrap().pop_front().unwrap_or(MockReply {
            status: 500,
            body: r#"{"error":"mock_ollama: chat queue empty"}"#.into(),
        });
    }
    if path.starts_with("/api/embed") {
        // 768-d zero vector — matches the `vec0` schema dim.
        let zeros: Vec<f32> = vec![0.0; 768];
        return MockReply::ok_json(serde_json::json!({ "embeddings": [zeros] }));
    }
    if path.starts_with("/api/tags") {
        return MockReply::ok_json(serde_json::json!({ "models": [] }));
    }
    MockReply {
        status: 404,
        body: r#"{"error":"mock_ollama: unknown path"}"#.into(),
    }
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
