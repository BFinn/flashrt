// SPDX-License-Identifier: Apache-2.0
//! GGUF metadata (not tensors): what the server needs from the model file, the vocabulary,
//! merges, token types, special ids and the chat template. Written from the GGUF v3 format
//! description.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};

use anyhow::{anyhow, bail, Result};

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
}

impl Value {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

struct R<T: Read>(T);

impl<T: Read> R<T> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; N];
        self.0.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        if n > 1 << 30 {
            bail!("gguf: string too long");
        }
        let mut b = vec![0u8; n];
        self.0.read_exact(&mut b)?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::Int(self.bytes::<1>()?[0] as i64),
            1 => Value::Int(self.bytes::<1>()?[0] as i8 as i64),
            2 => Value::Int(u16::from_le_bytes(self.bytes()?) as i64),
            3 => Value::Int(i16::from_le_bytes(self.bytes()?) as i64),
            4 => Value::Int(self.u32()? as i64),
            5 => Value::Int(i32::from_le_bytes(self.bytes()?) as i64),
            6 => Value::Float(f32::from_le_bytes(self.bytes()?) as f64),
            7 => Value::Bool(self.bytes::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                if n > 1 << 28 {
                    bail!("gguf: array too long");
                }
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Arr(v)
            }
            10 => Value::Int(self.u64()? as i64),
            11 => Value::Int(i64::from_le_bytes(self.bytes()?)),
            12 => Value::Float(f64::from_le_bytes(self.bytes()?)),
            _ => bail!("gguf: unknown value type {ty}"),
        })
    }
}

/// Reads the metadata key/value section of a GGUF file (shard 1 of a split model).
pub fn read_metadata(path: &str) -> Result<HashMap<String, Value>> {
    let f = File::open(path).map_err(|e| anyhow!("{path}: {e}"))?;
    let mut r = R(BufReader::with_capacity(1 << 20, f));
    if &r.bytes::<4>()? != b"GGUF" {
        bail!("{path}: not a GGUF file");
    }
    let version = r.u32()?;
    if version < 2 {
        bail!("{path}: GGUF version {version} is not supported");
    }
    let _n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    let mut kv = HashMap::new();
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        kv.insert(key, r.value(ty)?);
    }
    Ok(kv)
}
