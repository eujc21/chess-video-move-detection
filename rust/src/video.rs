//! Video decoding through an `ffmpeg` subprocess that streams raw RGB frames.
//!
//! Only the frames we actually need are decoded to RGB and piped: ffmpeg's
//! `select` filter drops everything else, which is far cheaper than
//! `cap.read()`-ing every frame as the Python version does.

use anyhow::{Context, Result, bail};
use std::io::Read;
use std::process::{Child, ChildStdout, Command, Stdio};

/// A decoded RGB24 frame.
pub struct Frame {
    /// 1-based frame number (matches the Python `frame_number`).
    pub number: u64,
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl Frame {
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        let i = (y * self.width + x) * 3;
        [self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]]
    }

    pub fn luma(&self, x: usize, y: usize) -> f64 {
        let [r, g, b] = self.pixel(x, y);
        0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64
    }
}

pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
}

pub fn probe(path: &str) -> Result<VideoInfo> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate:stream_side_data=rotation",
            "-of",
            "default=noprint_wrappers=1",
            path,
        ])
        .output()
        .context("failed to run ffprobe (is ffmpeg installed and on PATH?)")?;
    if !out.status.success() {
        bail!("ffprobe failed for {path}: {}", String::from_utf8_lossy(&out.stderr));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let (mut w, mut h, mut fps, mut rotation) = (0usize, 0usize, 0f64, 0i64);
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else { continue };
        match k {
            "width" => w = v.parse().unwrap_or(0),
            "height" => h = v.parse().unwrap_or(0),
            "avg_frame_rate" | "r_frame_rate" if fps == 0.0 => fps = parse_rate(v),
            "rotation" => rotation = v.parse::<f64>().map(|r| r as i64).unwrap_or(0),
            _ => {}
        }
    }
    if w == 0 || h == 0 || fps <= 0.0 {
        bail!("could not read video dimensions / fps for {path}");
    }
    // ffmpeg auto-rotates on decode, so report post-rotation dimensions.
    if rotation.rem_euclid(180) == 90 {
        std::mem::swap(&mut w, &mut h);
    }
    Ok(VideoInfo { width: w, height: h, fps })
}

fn parse_rate(v: &str) -> f64 {
    match v.split_once('/') {
        Some((n, d)) => {
            let (n, d): (f64, f64) = (n.parse().unwrap_or(0.0), d.parse().unwrap_or(0.0));
            if d > 0.0 { n / d } else { 0.0 }
        }
        None => v.parse().unwrap_or(0.0),
    }
}

/// Streams frame 1 plus every frame whose 1-based number is a multiple of
/// `interval` or of `stride`.
pub struct FrameReader {
    child: Child,
    stdout: ChildStdout,
    info: VideoInfo,
    interval: u64,
    stride: u64,
    last: u64,
}

impl FrameReader {
    pub fn open(path: &str, interval: u64, stride: u64) -> Result<Self> {
        let info = probe(path)?;
        let (interval, stride) = (interval.max(1), stride.max(1));
        // 0-based ffmpeg index n corresponds to 1-based frame n + 1.
        let select = format!("select='eq(n\\,0)+not(mod(n+1\\,{interval}))+not(mod(n+1\\,{stride}))'");
        let mut child = Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i", path, "-vf", &select, "-vsync", "0"])
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdout(Stdio::piped())
            .spawn()
            .context("failed to spawn ffmpeg (is it installed and on PATH?)")?;
        let stdout = child.stdout.take().context("ffmpeg stdout unavailable")?;
        Ok(Self { child, stdout, info, interval, stride, last: 0 })
    }

    pub fn next_frame(&mut self) -> Result<Option<Frame>> {
        let (w, h) = (self.info.width, self.info.height);
        let mut rgb = vec![0u8; w * h * 3];
        let mut filled = 0;
        while filled < rgb.len() {
            let n = self.stdout.read(&mut rgb[filled..])?;
            if n == 0 {
                if filled != 0 {
                    log::warn!("truncated trailing frame ignored");
                }
                return Ok(None);
            }
            filled += n;
        }
        let number = next_selected(self.last, self.interval, self.stride);
        self.last = number;
        Ok(Some(Frame { number, width: w, height: h, rgb }))
    }
}

/// The selected frame number following `last` (0 = before the first frame).
fn next_selected(last: u64, interval: u64, stride: u64) -> u64 {
    if last == 0 {
        return 1;
    }
    let next_multiple = |k: u64| (last / k + 1) * k;
    next_multiple(interval).min(next_multiple(stride))
}

impl Drop for FrameReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::next_selected;

    fn first(n: usize, interval: u64, stride: u64) -> Vec<u64> {
        let mut v = vec![];
        let mut last = 0;
        for _ in 0..n {
            last = next_selected(last, interval, stride);
            v.push(last);
        }
        v
    }

    #[test]
    fn selection_sequence() {
        assert_eq!(first(6, 29, 6), vec![1, 6, 12, 18, 24, 29]);
        assert_eq!(first(4, 1, 1), vec![1, 2, 3, 4]);
        assert_eq!(first(4, 30, 10), vec![1, 10, 20, 30]);
    }
}
