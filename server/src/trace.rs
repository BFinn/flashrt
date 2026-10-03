// SPDX-License-Identifier: Apache-2.0
//! Opt-in request traces (`--trace-dir DIR`): one JSON line per generation, in a file per UTC day
//! (`trace-YYYY-MM-DD.jsonl`). A line holds what was asked (the request as the server normalized
//! it, the sampling actually used, the rendered prompt and its token count), what came out as the
//! server parsed it (reasoning, text, tool calls, and tool-call text that did not parse), and the
//! engine's figures (its done event: reuse, timings, expert-cache hits, drafts). Traces hold whole
//! conversations, so the directory is created 0700 and the files 0600. Writing never fails a
//! request: an error is logged and the line dropped. Old files are not removed.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde_json::Value;

pub struct Tracer {
    dir: PathBuf,
    file: Mutex<Option<(String, File)>>,   // the current day and its open file
}

/// (year, month, day) of the civil date `days` after 1970-01-01 (H. Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// The current UTC time as (day "YYYY-MM-DD", timestamp "YYYY-MM-DDTHH:MM:SS.mmmZ").
pub fn now_utc() -> (String, String) {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let (secs, ms) = (t.as_secs() as i64, t.subsec_millis());
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let s = secs.rem_euclid(86_400);
    let day = format!("{y:04}-{m:02}-{d:02}");
    let ts = format!("{day}T{:02}:{:02}:{:02}.{ms:03}Z", s / 3600, s / 60 % 60, s % 60);
    (day, ts)
}

impl Tracer {
    pub fn new(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating the trace directory {}", dir.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
                .with_context(|| format!("restricting the trace directory {}", dir.display()))?;
        }
        Ok(Self { dir: dir.to_path_buf(), file: Mutex::new(None) })
    }

    fn open(&self, day: &str) -> std::io::Result<File> {
        let mut o = OpenOptions::new();
        o.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o600);
        }
        o.open(self.dir.join(format!("trace-{day}.jsonl")))
    }

    /// Appends one record as a line, stamped with the time it is written.
    pub fn write(&self, mut record: Value) {
        let (day, ts) = now_utc();
        if let Some(m) = record.as_object_mut() {
            m.insert("ts".into(), Value::String(ts));
        }
        let mut line = record.to_string();
        line.push('\n');
        let mut guard = self.file.lock().unwrap_or_else(|e| e.into_inner());
        let result = (|| {
            if guard.as_ref().is_none_or(|(d, _)| *d != day) {
                *guard = Some((day.clone(), self.open(&day)?));
            }
            let (_, f) = guard.as_mut().expect("opened above");
            f.write_all(line.as_bytes())?;
            f.flush()
        })();
        if let Err(e) = result {
            *guard = None;   // reopen on the next write
            tracing::warn!("trace write to {} failed: {e}", self.dir.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn civil_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
        assert_eq!(civil_from_days(20_729), (2026, 10, 3));
    }

    #[test]
    fn writes_private_lines_per_day() {
        let dir = std::env::temp_dir().join(format!("flashrt-trace-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let t = Tracer::new(&dir).unwrap();
        t.write(json!({"id": "r0", "output": {"content": "a\nb"}}));
        t.write(json!({"id": "r1"}));
        let (day, _) = now_utc();
        let path = dir.join(format!("trace-{day}.jsonl"));
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["output"]["content"], "a\nb");
        assert!(lines[1]["ts"].as_str().is_some_and(|s| s.starts_with(&day) && s.ends_with('Z')));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
            assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
