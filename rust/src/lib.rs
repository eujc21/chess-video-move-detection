//! Chess move extraction from video: board/piece/hand detection with YOLO
//! models (ONNX Runtime) and legal-move tracking.

pub mod board;
pub mod geometry;
pub mod notation;
pub mod pieces;
pub mod pipeline;
pub mod video;
pub mod yolo;

use geometry::Point;

/// Parses board corners given as `"x,y x,y x,y x,y"` (TL, TR, BR, BL).
pub fn parse_corners(s: &str) -> Result<[Point; 4], String> {
    let pts: Vec<Point> = s
        .split_whitespace()
        .map(|p| {
            let (x, y) = p.split_once(',').ok_or("expected x,y")?;
            Ok(Point::new(x.parse().map_err(|_| "bad x")?, y.parse().map_err(|_| "bad y")?))
        })
        .collect::<Result<_, &str>>()?;
    pts.try_into().map_err(|_| "expected four corners".to_string())
}

/// Default directory for CoreML's compiled-model cache.
pub fn default_cache_dir() -> Option<String> {
    let base = if cfg!(target_os = "macos") {
        std::env::var("HOME").ok().map(|h| format!("{h}/Library/Caches"))
    } else {
        std::env::var("XDG_CACHE_HOME").ok().or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.cache")))
    }?;
    let dir = format!("{base}/chess-video-moves");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}
