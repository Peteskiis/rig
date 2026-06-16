//! OpenAI API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::openai};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openai::Client::new("YOUR_API_KEY")?;
//!
//! let model = client.completion_model(openai::GPT_5_2);
//! # Ok(())
//! # }
//! ```
pub mod client;
pub mod completion;
pub mod embedding;
pub mod model_listing;
pub mod responses_api;

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
#[cfg(feature = "image")]
pub use image_generation::*;

pub mod transcription;

pub use client::*;
pub use completion::*;
pub use embedding::*;
pub use model_listing::*;

/// Recursively ensures all object schemas in a JSON schema respect OpenAI structured output restrictions.
/// Nested arrays, schema $defs, object properties and enums should be handled through this method
/// Sanitize a JSON schema for OpenAI's **strict** mode (structured outputs,
/// which OpenAI requires to be strict): forces `additionalProperties: false`
/// and `required` = every property, on top of the always-on cleanup (`$ref`
/// sibling-strip, `oneOf`→`anyOf`, recursion).
pub(crate) fn sanitize_schema(schema: &mut serde_json::Value) {
    sanitize_schema_impl(schema, true);
}

/// Like [`sanitize_schema`] but WITHOUT the strict-mode rewrites — for proxied
/// function tools where the caller can't know the client wants strict tool
/// calling (cluster gateway patch, agent-api#133). Skips forcing
/// `additionalProperties: false` and the `required` = all-properties rewrite, so
/// OpenAI's lenient (non-strict) tool validation accepts open maps
/// (`additionalProperties` with no fixed `properties`) and property-less objects
/// instead of 400ing `invalid_function_parameters`. The benign cleanup (sibling
/// strip, `oneOf`→`anyOf`, recursion) still runs.
pub(crate) fn sanitize_schema_lenient(schema: &mut serde_json::Value) {
    sanitize_schema_impl(schema, false);
}

fn sanitize_schema_impl(schema: &mut serde_json::Value, strict: bool) {
    use serde_json::Value;

    if let Value::Object(obj) = schema {
        // OpenAI does not allow sibling keywords next to $ref (e.g. "description").
        // Strip everything except $ref so the reference is the sole key.
        if obj.contains_key("$ref") {
            obj.retain(|k, _| k == "$ref");
            return;
        }

        // The two strict-only rewrites — skipped in lenient mode so a valid
        // non-strict schema (open map / property-less object) is left intact.
        if strict {
            let is_object_schema = obj.get("type") == Some(&Value::String("object".to_string()))
                || obj.contains_key("properties");

            // Required by OpenAI's Responses API when using strict mode.
            // Source: https://platform.openai.com/docs/guides/structured-outputs#additionalproperties-false-must-always-be-set-in-objects
            if is_object_schema && !obj.contains_key("additionalProperties") {
                obj.insert("additionalProperties".to_string(), Value::Bool(false));
            }

            // Also required by OpenAI's Responses API in strict mode.
            // Source: https://platform.openai.com/docs/guides/structured-outputs#all-fields-must-be-required
            if let Some(Value::Object(properties)) = obj.get("properties") {
                let prop_keys = properties.keys().cloned().map(Value::String).collect();
                obj.insert("required".to_string(), Value::Array(prop_keys));
            }
        }

        if let Some(defs) = obj.get_mut("$defs")
            && let Value::Object(defs_obj) = defs
        {
            for (_, def_schema) in defs_obj.iter_mut() {
                sanitize_schema_impl(def_schema, strict);
            }
        }

        if let Some(properties) = obj.get_mut("properties")
            && let Value::Object(props) = properties
        {
            for (_, prop_value) in props.iter_mut() {
                sanitize_schema_impl(prop_value, strict);
            }
        }

        if let Some(items) = obj.get_mut("items") {
            sanitize_schema_impl(items, strict);
        }

        // OpenAI doesn't support oneOf so we need to switch this to anyOf
        if let Some(one_of) = obj.remove("oneOf") {
            // If `anyOf` already exists, merge arrays. If not, insert new.
            match obj.get_mut("anyOf") {
                Some(Value::Array(existing)) => {
                    if let Value::Array(mut incoming) = one_of {
                        existing.append(&mut incoming);
                    }
                }
                _ => {
                    obj.insert("anyOf".to_string(), one_of);
                }
            }
        }

        // should handle Enums (anyOf/oneOf)
        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(variants) = obj.get_mut(key)
                && let Value::Array(variants_array) = variants
            {
                for variant in variants_array.iter_mut() {
                    sanitize_schema_impl(variant, strict);
                }
            }
        }
    }
}

#[cfg(feature = "audio")]
pub use audio_generation::{TTS_1, TTS_1_HD};

pub use streaming::*;
pub use transcription::*;

#[cfg(test)]
mod tests {
    use super::sanitize_schema;
    use serde_json::json;

    #[test]
    fn test_sanitize_strips_ref_sibling_keywords() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "location": {
                    "$ref": "#/$defs/Location",
                    "description": "The user's location"
                }
            },
            "$defs": {
                "Location": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" },
                        "state": { "type": "string" }
                    }
                }
            }
        });

        sanitize_schema(&mut schema);

        // $ref node should only contain "$ref", no "description"
        let location = &schema["properties"]["location"];
        assert_eq!(location, &json!({ "$ref": "#/$defs/Location" }));

        // The referenced $def should still be fully sanitized
        let location_def = &schema["$defs"]["Location"];
        assert_eq!(location_def["additionalProperties"], json!(false));
        assert!(location_def["required"].as_array().is_some());
    }

    #[test]
    fn test_sanitize_adds_additional_properties_false() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });

        sanitize_schema(&mut schema);

        assert_eq!(schema["additionalProperties"], json!(false));
    }

    #[test]
    fn test_sanitize_marks_all_properties_required() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "a": { "type": "string" },
                "b": { "type": "number" }
            }
        });

        sanitize_schema(&mut schema);

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("a")));
        assert!(required.contains(&json!("b")));
        assert_eq!(required.len(), 2);
    }

    #[test]
    fn test_sanitize_converts_one_of_to_any_of() {
        let mut schema = json!({
            "oneOf": [
                { "type": "string" },
                { "type": "number" }
            ]
        });

        sanitize_schema(&mut schema);

        assert!(schema.get("oneOf").is_none());
        assert!(schema["anyOf"].as_array().is_some());
    }

    #[test]
    fn test_sanitize_recurses_into_nested_objects() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {
                        "value": { "type": "string" }
                    }
                }
            }
        });

        sanitize_schema(&mut schema);

        assert_eq!(
            schema["properties"]["inner"]["additionalProperties"],
            json!(false)
        );
        let inner_required = schema["properties"]["inner"]["required"]
            .as_array()
            .unwrap();
        assert!(inner_required.contains(&json!("value")));
    }

    // ---- lenient (non-strict) tool sanitize — cluster gateway patch (agent-api#133) ----

    #[test]
    fn lenient_preserves_open_map_and_skips_required_rewrite() {
        use super::sanitize_schema_lenient;
        // The browser_drop-style shape that strict mode rejects: an open
        // string-map `data` that is optional (`required: ["target"]`). Strict
        // sanitize would force `additionalProperties: false` and rewrite
        // `required` to every key (adding `data`), which OpenAI 400s. Lenient
        // must leave both alone.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "target": { "type": "string" },
                "data": { "type": "object", "additionalProperties": { "type": "string" } }
            },
            "required": ["target"]
        });

        sanitize_schema_lenient(&mut schema);

        // No forced additionalProperties: false on the root...
        assert_eq!(schema.get("additionalProperties"), None);
        // ...the open map's `additionalProperties` is preserved (not clobbered)...
        assert_eq!(
            schema["properties"]["data"]["additionalProperties"],
            json!({ "type": "string" })
        );
        // ...and `required` is untouched (still just `target`, NOT every key).
        assert_eq!(schema["required"], json!(["target"]));
    }

    #[test]
    fn lenient_still_runs_benign_cleanup() {
        use super::sanitize_schema_lenient;
        // oneOf→anyOf and $ref sibling-strip still apply in lenient mode.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "choice": { "oneOf": [ { "type": "string" }, { "type": "number" } ] },
                "ref": { "$ref": "#/$defs/X", "description": "drop me" }
            }
        });

        sanitize_schema_lenient(&mut schema);

        assert!(schema["properties"]["choice"].get("oneOf").is_none());
        assert!(schema["properties"]["choice"].get("anyOf").is_some());
        assert_eq!(
            schema["properties"]["ref"],
            json!({ "$ref": "#/$defs/X" }),
            "$ref siblings stripped even in lenient mode"
        );
    }

    #[test]
    fn strict_still_forces_rewrites() {
        // The strict path (structured outputs) is unchanged by the split.
        let mut schema = json!({
            "type": "object",
            "properties": { "a": { "type": "string" } }
        });
        sanitize_schema(&mut schema);
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(schema["required"], json!(["a"]));
    }
}
