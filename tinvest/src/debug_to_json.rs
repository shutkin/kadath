use heuristic_debug_parser::{OrNonExhaustive, Term, Value};

pub fn debug_value_to_json(val: Value) -> serde_json::Value {
    match val {
        Value::Term(term) => match term {
            Term::Ident(s) => serde_json::Value::String(s),
            Term::String(s) => serde_json::Value::String(s),
            Term::UnquotedRawString(n) => {
                if n.to_lowercase() == "null" {
                    serde_json::Value::Null
                } else if let Ok(i) = n.parse::<i64>() {
                    serde_json::Value::Number(i.into())
                } else if let Ok(f) = n.parse::<f64>() {
                    if let Some(num) = serde_json::Number::from_f64(f) {
                        serde_json::Value::Number(num)
                    } else {
                        serde_json::Value::String(n)
                    }
                } else if let Ok(b) = n.parse::<bool>() {
                    serde_json::Value::Bool(b)
                } else {
                    serde_json::Value::String(n)
                }
            }
        },

        Value::List(vec) => {
            let json_arr = vec.values.into_iter().map(debug_value_to_json).collect();
            serde_json::Value::Array(json_arr)
        }

        Value::Map(map) => {
            let json_obj = map.values
                .into_iter()
                .map(|key_value| (value_to_string(key_value.key), debug_value_to_json(key_value.value)))
                .collect();
            serde_json::Value::Object(json_obj)
        }

        Value::Struct(map) => {
            let json_obj = map.values
                .into_iter()
                .filter_map(|kv| {
                    match kv {
                        OrNonExhaustive::Value(ident_value) => {
                            Some((ident_value.ident, debug_value_to_json(ident_value.value)))
                        }
                        OrNonExhaustive::NonExhaustive => {
                            None
                        }
                    }
                })
                .collect();
            serde_json::Value::Object(json_obj)
        }

        Value::Set(set) => {
            let json_arr = set.values.into_iter().map(debug_value_to_json).collect();
            serde_json::Value::Array(json_arr)
        }

        Value::Tuple(tuple) => {
            let json_arr = tuple.values.into_iter().map(debug_value_to_json).collect();
            serde_json::Value::Array(json_arr)
        }
    }
}

fn value_to_string(val: Value) -> String {
    if let Value::Term(term) = val {
        match term {
            Term::Ident(s) => s,
            Term::String(s) => s,
            Term::UnquotedRawString(s) => s,
        }
    } else {
        String::new()
    }
}