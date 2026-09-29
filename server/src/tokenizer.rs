// SPDX-License-Identifier: Apache-2.0
//! Byte-level BPE tokenizer built from the GGUF's vocabulary and merges (GPT-2 style byte
//! mapping, merges applied lowest rank first), with the pre-tokenizer pattern of llama.cpp's
//! QWEN35 pre-type (src/llama-vocab.cpp, MIT). Special tokens (control and user-defined) are
//! matched as whole strings before pre-tokenization. `flashrt-server --check-tokenizer` compares
//! it with a llama.cpp tokenization.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Result};
use fancy_regex::Regex;

use crate::gguf::Value;

// llama.cpp src/llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_QWEN35 (MIT)
const QWEN35_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    pub n_vocab: usize,
    pub eos: u32,
    token_bytes: Vec<Vec<u8>>,              // id -> the bytes it stands for
    byte_token: [u32; 256],                 // the token of each single byte
    merges: HashMap<(u32, u32), (u32, u32)>, // pair -> (rank, merged id)
    specials: Vec<(String, u32)>,           // longest first
    special_ids: Vec<bool>,
    control_ids: Vec<bool>,
    pretok: Regex,
    cache: Mutex<HashMap<String, Vec<u32>>>,
}

// GPT-2's reversible byte <-> unicode mapping
fn byte_to_char() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    map
}

impl Tokenizer {
    pub fn from_gguf(kv: &HashMap<String, Value>) -> Result<Self> {
        let model = kv.get("tokenizer.ggml.model").and_then(Value::as_str).unwrap_or("");
        if model != "gpt2" {
            bail!("tokenizer model '{model}' is not supported (byte-level BPE only)");
        }
        // the pre-tokenizer pattern is llama.cpp's for this type; another type tokenizes wrongly
        let pre = kv.get("tokenizer.ggml.pre").and_then(Value::as_str).unwrap_or("");
        if pre != "qwen35" {
            bail!("pre-tokenizer '{pre}' is not supported (only qwen35, whose pattern this tokenizer implements)");
        }
        let tokens: Vec<&str> = kv
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_arr)
            .ok_or_else(|| anyhow!("gguf: no tokenizer.ggml.tokens"))?
            .iter()
            .map(|v| v.as_str().unwrap_or(""))
            .collect();
        let types: Vec<i64> = kv
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_arr)
            .map(|a| a.iter().map(|v| v.as_int().unwrap_or(1)).collect())
            .unwrap_or_else(|| vec![1; tokens.len()]);
        let b2c = byte_to_char();
        let mut c2b: HashMap<char, u8> = HashMap::new();
        for (b, c) in b2c.iter().enumerate() {
            c2b.insert(*c, b as u8);
        }
        let mut id_of: HashMap<&str, u32> = HashMap::with_capacity(tokens.len());
        let mut token_bytes = Vec::with_capacity(tokens.len());
        let mut specials = Vec::new();
        let mut special_ids = vec![false; tokens.len()];
        let mut control_ids = vec![false; tokens.len()];
        for (i, t) in tokens.iter().enumerate() {
            id_of.insert(t, i as u32);
            let special = types.get(i).is_some_and(|&ty| ty == 3 || ty == 4);   // control, user-defined
            control_ids[i] = types.get(i) == Some(&3);
            if special {
                specials.push((t.to_string(), i as u32));
                special_ids[i] = true;
                token_bytes.push(t.as_bytes().to_vec());
            } else {
                let mut bytes = Vec::with_capacity(t.len());
                for ch in t.chars() {
                    match c2b.get(&ch) {
                        Some(&b) => bytes.push(b),
                        None => bytes.extend_from_slice(ch.to_string().as_bytes()),
                    }
                }
                token_bytes.push(bytes);
            }
        }
        specials.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
        let mut byte_token = [0u32; 256];
        for b in 0..256 {
            byte_token[b] = *id_of
                .get(b2c[b].to_string().as_str())
                .ok_or_else(|| anyhow!("tokenizer: byte {b} has no token"))?;
        }
        let mut merges = HashMap::new();
        if let Some(arr) = kv.get("tokenizer.ggml.merges").and_then(Value::as_arr) {
            for (rank, m) in arr.iter().enumerate() {
                let Some(s) = m.as_str() else { continue };
                let Some((a, b)) = s.split_once(' ') else { continue };
                let (Some(&ia), Some(&ib)) = (id_of.get(a), id_of.get(b)) else { continue };
                let joined = format!("{a}{b}");
                let Some(&im) = id_of.get(joined.as_str()) else { continue };
                merges.entry((ia, ib)).or_insert((rank as u32, im));
            }
        }
        let eos = kv
            .get("tokenizer.ggml.eos_token_id")
            .and_then(Value::as_int)
            .filter(|&e| e >= 0 && (e as usize) < tokens.len())
            .ok_or_else(|| anyhow!("gguf: no valid tokenizer.ggml.eos_token_id"))? as u32;
        Ok(Self {
            n_vocab: tokens.len(),
            eos,
            token_bytes,
            byte_token,
            merges,
            specials,
            special_ids,
            control_ids,
            pretok: Regex::new(QWEN35_PATTERN)?,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.specials.iter().find(|(s, _)| s == text).map(|(_, id)| *id)
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.get(id as usize).copied().unwrap_or(false)
    }

    /// A control token (GGUF token type 3, such as <|im_start|>): structure, never output text.
    pub fn is_control(&self, id: u32) -> bool {
        self.control_ids.get(id as usize).copied().unwrap_or(false)
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.token_bytes.get(id as usize).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Tokenizes text; with `specials`, the special tokens' strings become their ids.
    pub fn encode(&self, text: &str, specials: bool) -> Vec<u32> {
        let mut out = Vec::new();
        let mut start = 0;
        if specials {
            let bytes = text.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'<' {
                    if let Some((s, id)) = self.specials.iter().find(|(s, _)| text[i..].starts_with(s.as_str())) {
                        self.encode_plain(&text[start..i], &mut out);
                        out.push(*id);
                        i += s.len();
                        start = i;
                        continue;
                    }
                }
                i += 1;
            }
        }
        self.encode_plain(&text[start..], &mut out);
        out
    }

    /// The words' ids, from the cache or merged. The cache lock is taken per word, so a long
    /// text does not hold up other requests' tokenization; words longer than CACHE_WORD bytes are
    /// not cached, which bounds the cache at CACHE_WORDS entries of that size.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        const CACHE_WORDS: usize = 200_000;
        const CACHE_WORD: usize = 256;
        for m in self.pretok.find_iter(text) {
            let Ok(m) = m else { continue };
            let word = m.as_str();
            if let Some(ids) = self.cache.lock().unwrap().get(word) {
                out.extend_from_slice(ids);
                continue;
            }
            let ids = self.bpe(word.as_bytes());
            out.extend_from_slice(&ids);
            if word.len() <= CACHE_WORD {
                let mut cache = self.cache.lock().unwrap();
                if cache.len() >= CACHE_WORDS {
                    cache.clear();
                }
                cache.insert(word.to_string(), ids);
            }
        }
    }

    /// Byte-level BPE: repeatedly merges the adjacent pair of lowest rank, the leftmost among
    /// equals. A heap of candidate pairs over a linked list of symbols makes it O(n log n) in the
    /// word's length (a pre-token can be a long run of letters); popping (rank, left index) gives
    /// the same order as scanning for the leftmost lowest rank.
    fn bpe(&self, word: &[u8]) -> Vec<u32> {
        const NONE: usize = usize::MAX;
        let n = word.len();
        let mut sym: Vec<u32> = word.iter().map(|&b| self.byte_token[b as usize]).collect();
        let mut next: Vec<usize> = (1..=n).map(|i| if i < n { i } else { NONE }).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| if i > 0 { i - 1 } else { NONE }).collect();
        let mut alive = vec![true; n];
        // (rank, left, right, left symbol, right symbol, merged id); stale entries are skipped
        type Candidate = Reverse<(u32, usize, usize, u32, u32, u32)>;
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<Candidate>, sym: &[u32], l: usize, r: usize| {
            if let Some(&(rank, id)) = self.merges.get(&(sym[l], sym[r])) {
                heap.push(Reverse((rank, l, r, sym[l], sym[r], id)));
            }
        };
        for i in 0..n.saturating_sub(1) {
            push(&mut heap, &sym, i, i + 1);
        }
        while let Some(Reverse((_, l, r, a, b, id))) = heap.pop() {
            if !alive[l] || !alive[r] || next[l] != r || sym[l] != a || sym[r] != b {
                continue;
            }
            sym[l] = id;
            alive[r] = false;
            next[l] = next[r];
            if next[r] != NONE {
                prev[next[r]] = l;
            }
            if prev[l] != NONE {
                push(&mut heap, &sym, prev[l], l);
            }
            if next[l] != NONE {
                push(&mut heap, &sym, l, next[l]);
            }
        }
        (0..n).filter(|&i| alive[i]).map(|i| sym[i]).collect()
    }
}

/// Turns a token stream back into text, holding back bytes of an incomplete UTF-8 sequence.
#[derive(Default)]
pub struct Decoder {
    pending: Vec<u8>,
}

impl Decoder {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    match e.error_len() {
                        None => {
                            // an incomplete sequence at the end: keep it for the next token
                            self.pending.drain(..valid);
                            return out;
                        }
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }

    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// A byte-level tokenizer for tests: ids 0..255 are the bytes, then <think> (256), </think>,
/// <tool_call>, </tool_call> (user-defined), <|im_start|> (260) and <|im_end|> (control).
#[cfg(test)]
pub fn test_tokenizer() -> Tokenizer {
    let b2c = byte_to_char();
    let mut tokens: Vec<Value> = (0..256).map(|b| Value::Str(b2c[b].to_string())).collect();
    let mut types = vec![Value::Int(1); 256];
    for (t, ty) in [("<think>", 4), ("</think>", 4), ("<tool_call>", 4), ("</tool_call>", 4), ("<|im_start|>", 3), ("<|im_end|>", 3)] {
        tokens.push(Value::Str(t.into()));
        types.push(Value::Int(ty));
    }
    let kv = HashMap::from([
        ("tokenizer.ggml.model".to_string(), Value::Str("gpt2".into())),
        ("tokenizer.ggml.pre".to_string(), Value::Str("qwen35".into())),
        ("tokenizer.ggml.tokens".to_string(), Value::Arr(tokens)),
        ("tokenizer.ggml.token_type".to_string(), Value::Arr(types)),
        ("tokenizer.ggml.eos_token_id".to_string(), Value::Int(261)),
    ]);
    Tokenizer::from_gguf(&kv).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes plus merged tokens over 'a', 'b' and 'c', with overlapping merges of several ranks.
    fn merges_tokenizer() -> Tokenizer {
        let b2c = byte_to_char();
        let mut tokens: Vec<Value> = (0..256).map(|b| Value::Str(b2c[b].to_string())).collect();
        let merged = ["aa", "ab", "bc", "ba", "aaa", "abc", "aab", "aaaaaa", "bcbc", "aba", "cab"];
        tokens.extend(merged.iter().map(|t| Value::Str(t.to_string())));
        let merges = ["a b", "a a", "b c", "b a", "aa a", "ab c", "aa b", "aaa aaa", "bc bc", "ab a", "c ab"];
        let kv = HashMap::from([
            ("tokenizer.ggml.model".to_string(), Value::Str("gpt2".into())),
            ("tokenizer.ggml.pre".to_string(), Value::Str("qwen35".into())),
            ("tokenizer.ggml.tokens".to_string(), Value::Arr(tokens)),
            ("tokenizer.ggml.merges".to_string(), Value::Arr(merges.iter().map(|m| Value::Str(m.to_string())).collect())),
            ("tokenizer.ggml.eos_token_id".to_string(), Value::Int(0)),
        ]);
        Tokenizer::from_gguf(&kv).unwrap()
    }

    /// The merge as first written: scan for the leftmost pair of lowest rank, merge, repeat.
    fn scan_bpe(tok: &Tokenizer, word: &[u8]) -> Vec<u32> {
        let mut parts: Vec<u32> = word.iter().map(|&b| tok.byte_token[b as usize]).collect();
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                if let Some(&(rank, id)) = tok.merges.get(&(parts[i], parts[i + 1])) {
                    if best.is_none_or(|(r, _, _)| rank < r) {
                        best = Some((rank, i, id));
                    }
                }
            }
            let Some((_, i, id)) = best else { break };
            parts[i] = id;
            parts.remove(i + 1);
        }
        parts
    }

    #[test]
    fn rejects_other_pre_tokenizers_and_a_missing_eos() {
        let base = |pre: &str, eos: Option<i64>| {
            let b2c = byte_to_char();
            let mut kv = HashMap::from([
                ("tokenizer.ggml.model".to_string(), Value::Str("gpt2".into())),
                ("tokenizer.ggml.pre".to_string(), Value::Str(pre.into())),
                ("tokenizer.ggml.tokens".to_string(), Value::Arr((0..256).map(|b| Value::Str(b2c[b].to_string())).collect())),
            ]);
            if let Some(e) = eos {
                kv.insert("tokenizer.ggml.eos_token_id".to_string(), Value::Int(e));
            }
            Tokenizer::from_gguf(&kv)
        };
        assert!(base("qwen35", Some(1)).is_ok());
        assert!(base("qwen2", Some(1)).is_err());
        assert!(base("qwen35", None).is_err());
        assert!(base("qwen35", Some(256)).is_err());
    }

    #[test]
    fn heap_bpe_matches_the_scan() {
        let tok = merges_tokenizer();
        let mut x = 0x9E3779B97F4A7C15u64;
        for len in 0..400 {
            for alphabet in [&b"ab"[..], b"abc", b"a", b"abcd"] {
                let w: Vec<u8> = (0..len % 97)
                    .map(|_| {
                        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        alphabet[(x >> 33) as usize % alphabet.len()]
                    })
                    .collect();
                assert_eq!(tok.bpe(&w), scan_bpe(&tok, &w), "word {:?}", String::from_utf8_lossy(&w));
            }
        }
        let long: Vec<u8> = (0..3000).map(|i| b"aababcaaab"[i % 10]).collect();
        assert_eq!(tok.bpe(&long), scan_bpe(&tok, &long));
        // a long run of letters (one pre-token): quick (the scan was O(n^2)), and the bytes survive
        let t = std::time::Instant::now();
        let word = vec![b'a'; 1 << 20];
        let ids = tok.bpe(&word);
        assert!(t.elapsed().as_secs_f64() < 2.0, "{:?}", t.elapsed());
        let back: Vec<u8> = ids.iter().flat_map(|&id| tok.token_bytes(id).to_vec()).collect();
        assert_eq!(back, word);
    }

    #[test]
    fn decoder_holds_incomplete_utf8() {
        let mut d = Decoder::default();
        let e = "é".as_bytes();
        assert_eq!(d.push(&e[..1]), "");
        assert_eq!(d.push(&e[1..]), "é");
        assert_eq!(d.push(b"ok \xF0\x9F"), "ok ");   // half an emoji held back
        assert_eq!(d.push(b"\x98\x80!"), "\u{1F600}!");
        assert_eq!(d.push(b"\xFFx"), "\u{FFFD}x");   // an invalid byte becomes U+FFFD
        d.push(b"\xE2\x82");
        assert_eq!(d.finish(), "\u{FFFD}");   // an unfinished sequence at the end
    }

    // With FLASHRT_TEST_MODEL set to the model's first GGUF shard: encoding then decoding
    // round-trips text, and special tokens map to single ids.
    #[test]
    fn model_round_trip() {
        let Ok(path) = std::env::var("FLASHRT_TEST_MODEL") else {
            eprintln!("FLASHRT_TEST_MODEL not set; skipped");
            return;
        };
        let kv = crate::gguf::read_metadata(&path).unwrap();
        let tok = Tokenizer::from_gguf(&kv).unwrap();
        for text in ["Hello, world! 1234 5678", "ünïcödé, 日本語, emoji 😀 and\ttabs\n\nnewlines", "  leading and trailing  "] {
            let ids = tok.encode(text, false);
            let mut d = Decoder::default();
            let mut back = String::new();
            for id in &ids {
                back.push_str(&d.push(tok.token_bytes(*id)));
            }
            back.push_str(&d.finish());
            assert_eq!(back, text);
        }
        let im = tok.token_id("<|im_start|>").expect("<|im_start|> is a special token");
        assert_eq!(tok.encode("<|im_start|>user", true)[0], im);
        assert!(tok.is_special(im));
    }
}
