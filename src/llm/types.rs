//! Types shared between the LLM client and the rest of the agent.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "system" | "user" | "assistant" | "tool"
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>, // base64-encoded image bytes
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>, // tool name when role = "tool"
}

impl ChatMessage {
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: text.into(),
            images: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: text.into(),
            images: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
    pub fn user_with_images(text: impl Into<String>, images: Vec<String>) -> Self {
        Self {
            role: "user".into(),
            content: text.into(),
            images: Some(images),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
    #[allow(dead_code)]
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: text.into(),
            images: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
    pub fn tool_result(name: impl Into<String>, content: impl Into<String>) -> Self {
        let name_s: String = name.into();
        Self {
            role: "tool".into(),
            content: content.into(),
            images: None,
            tool_calls: None,
            tool_call_id: None,
            name: Some(name_s),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default = "default_type")]
    pub r#type: String,
    pub function: ToolFunctionCall,
}
fn default_type() -> String {
    "function".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    #[serde(rename = "type")]
    pub r#type: String, // "function"
    pub function: ToolFunctionSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunctionSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON schema
}
