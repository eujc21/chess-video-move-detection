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

/// Where frames come from.
#[derive(Debug, Clone)]
pub enum Source {
    /// A video file. `realtime` paces decoding at the video's frame rate (for live-style preview).
    File { path: String, realtime: bool },
    /// A capture device: an AVFoundation index or name on macOS, `/dev/videoN`
    /// on Linux, a DirectShow device name on Windows.
    Camera { device: String, width: usize, height: usize, fps: f64 },
}

impl FrameReader {
    /// Opens a file and decodes only frame 1 and multiples of `interval` or `stride`.
    pub fn open(path: &str, interval: u64, stride: u64) -> Result<Self> {
        let info = probe(path)?;
        let (interval, stride) = (interval.max(1), stride.max(1));
        // 0-based ffmpeg index n corresponds to 1-based frame n + 1.
        let select = format!("select='eq(n\\,0)+not(mod(n+1\\,{interval}))+not(mod(n+1\\,{stride}))'");
        let mut args = hwaccel_args();
        args.extend(["-i".into(), path.into(), "-vf".into(), select]);
        args.extend(passthrough_args().map(String::from));
        Self::spawn(args, info, interval, stride)
    }

    /// Opens a file or camera and yields every frame.
    pub fn open_source(source: &Source) -> Result<Self> {
        let (mut args, info) = match source {
            Source::File { path, realtime } => {
                let mut args = hwaccel_args();
                if *realtime {
                    args.push("-re".into());
                }
                args.extend(["-i".into(), path.clone()]);
                (args, probe(path)?)
            }
            Source::Camera { device, width, height, fps } => {
                let size = format!("{width}x{height}");
                let rate = format!("{fps}");
                let args: Vec<String> = if cfg!(target_os = "macos") {
                    vec!["-f", "avfoundation", "-framerate", &rate, "-video_size", &size, "-i", device]
                        .into_iter()
                        .map(String::from)
                        .collect()
                } else if cfg!(target_os = "windows") {
                    let input = format!("video={device}");
                    vec!["-f", "dshow", "-framerate", &rate, "-video_size", &size, "-i", &input]
                        .into_iter()
                        .map(String::from)
                        .collect()
                } else {
                    vec!["-f", "v4l2", "-framerate", &rate, "-video_size", &size, "-i", device]
                        .into_iter()
                        .map(String::from)
                        .collect()
                };
                (args, VideoInfo { width: *width, height: *height, fps: *fps })
            }
        };
        // Guarantee the frame size we read, whatever the device delivers.
        args.extend(["-vf".into(), format!("scale={}:{}", info.width, info.height)]);
        Self::spawn(args, info, 1, 1)
    }

    fn spawn(input_args: Vec<String>, info: VideoInfo, interval: u64, stride: u64) -> Result<Self> {
        let mut child = Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin"])
            .args(&input_args)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .spawn()
            .context("failed to spawn ffmpeg (is it installed and on PATH?)")?;
        let stdout = child.stdout.take().context("ffmpeg stdout unavailable")?;
        Ok(Self { child, stdout, info, interval, stride, last: 0 })
    }

    pub fn info(&self) -> &VideoInfo {
        &self.info
    }

    pub fn next_frame(&mut self) -> Result<Option<Frame>> {
        let (w, h) = (self.info.width, self.info.height);
        let mut rgb = vec![0u8; w * h * 3];
        let mut filled = 0;
        while filled < rgb.len() {
            let n = self.stdout.read(&mut rgb[filled..])?;
            if n == 0 {
                let status = self.child.wait()?;
                if !status.success() {
                    bail!("ffmpeg failed ({status}); see its message above");
                }
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

/// Keeps every selected frame exactly once (no duplicates to fill gaps). ffmpeg
/// 5.1 replaced `-vsync` with `-fps_mode`, and ffmpeg 8 removed `-vsync`.
fn passthrough_args() -> [&'static str; 2] {
    static MODERN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let modern = *MODERN.get_or_init(|| {
        let out = Command::new("ffmpeg").arg("-version").stdin(Stdio::null()).output();
        let text = out.map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
        ffmpeg_version(&text).is_none_or(|v| v >= (5, 1))
    });
    if modern { ["-fps_mode", "passthrough"] } else { ["-vsync", "0"] }
}

/// Parses `(major, minor)` from `ffmpeg -version` output, e.g. "ffmpeg version 6.1.1-3ubuntu5"
/// or "ffmpeg version n7.1". Git builds ("N-12345-g…") have no version and return `None`.
pub fn ffmpeg_version(text: &str) -> Option<(u32, u32)> {
    let version = text.split_whitespace().nth(2)?.trim_start_matches('n');
    let mut parts = version.split(|c: char| !c.is_ascii_digit());
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().and_then(|m| m.parse().ok()).unwrap_or(0);
    Some((major, minor))
}

/// Hardware video decoding: VideoToolbox (Apple media engine) on macOS.
fn hwaccel_args() -> Vec<String> {
    if cfg!(target_os = "macos") { vec!["-hwaccel".into(), "videotoolbox".into()] } else { Vec::new() }
}

/// Lists capture devices as `(device, label)` pairs using ffmpeg / the OS.
pub fn list_cameras() -> Vec<(String, String)> {
    // Linux, FreeBSD (webcamd) and other Unixes expose V4L2 devices as /dev/videoN.
    if !cfg!(any(target_os = "macos", target_os = "windows")) {
        let mut devs: Vec<(String, String)> = std::fs::read_dir("/dev")
            .map(|d| {
                d.flatten()
                    .map(|e| e.path().to_string_lossy().into_owned())
                    .filter(|p| p.starts_with("/dev/video"))
                    .map(|p| (p.clone(), p))
                    .collect()
            })
            .unwrap_or_default();
        devs.sort();
        return devs;
    }
    let args: &[&str] = if cfg!(target_os = "macos") {
        &["-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    } else {
        &["-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    };
    let Ok(out) = Command::new("ffmpeg").args(args).stdin(Stdio::null()).output() else { return Vec::new() };
    parse_device_list(&String::from_utf8_lossy(&out.stderr))
}

/// Parses `ffmpeg -list_devices` output (AVFoundation or DirectShow).
pub fn parse_device_list(text: &str) -> Vec<(String, String)> {
    let mut devices = Vec::new();
    let mut in_video = true;
    for line in text.lines() {
        if line.contains("AVFoundation audio devices") {
            in_video = false;
        } else if line.contains("AVFoundation video devices") {
            in_video = true;
        }
        let Some(msg) = line.split_once("] ").map(|(_, m)| m.trim()) else { continue };
        if msg.starts_with('[') && in_video {
            // AVFoundation: "[0] FaceTime HD Camera"
            if let Some((idx, name)) = msg[1..].split_once("] ") {
                devices.push((idx.to_string(), format!("{idx}: {name}")));
            }
        } else if msg.starts_with('"') && msg.ends_with("(video)") {
            // DirectShow: "\"Integrated Camera\" (video)"
            let name = msg.trim_end_matches("(video)").trim().trim_matches('"').to_string();
            devices.push((name.clone(), name));
        }
    }
    devices
}

/// The selected frame number following `last` (0 = before the first frame).
pub fn next_selected(last: u64, interval: u64, stride: u64) -> u64 {
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
    fn parses_device_lists() {
        let mac = "[AVFoundation indev @ 0x1] AVFoundation video devices:\n[AVFoundation indev @ 0x1] [0] FaceTime HD Camera\n[AVFoundation indev @ 0x1] [1] Capture screen 0\n[AVFoundation indev @ 0x1] AVFoundation audio devices:\n[AVFoundation indev @ 0x1] [0] MacBook Pro Microphone";
        assert_eq!(
            super::parse_device_list(mac),
            vec![
                ("0".to_string(), "0: FaceTime HD Camera".to_string()),
                ("1".to_string(), "1: Capture screen 0".to_string())
            ]
        );
        let win = "[dshow @ 0x1] \"Integrated Camera\" (video)\n[dshow @ 0x1]   Alternative name \"@device_pnp\"\n[dshow @ 0x1] \"Microphone\" (audio)";
        assert_eq!(
            super::parse_device_list(win),
            vec![("Integrated Camera".to_string(), "Integrated Camera".to_string())]
        );
    }

    #[test]
    fn parses_ffmpeg_versions() {
        use super::ffmpeg_version;
        assert_eq!(ffmpeg_version("ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023"), Some((6, 1)));
        assert_eq!(ffmpeg_version("ffmpeg version n7.1 Copyright"), Some((7, 1)));
        assert_eq!(ffmpeg_version("ffmpeg version 8.0 Copyright"), Some((8, 0)));
        assert_eq!(ffmpeg_version("ffmpeg version 4.4.2-0ubuntu0.22.04.1"), Some((4, 4)));
        assert_eq!(ffmpeg_version("ffmpeg version N-118000-gabcdef"), None);
    }

    #[test]
    fn selection_sequence() {
        assert_eq!(first(6, 29, 6), vec![1, 6, 12, 18, 24, 29]);
        assert_eq!(first(4, 1, 1), vec![1, 2, 3, 4]);
        assert_eq!(first(4, 30, 10), vec![1, 10, 20, 30]);
    }
}
