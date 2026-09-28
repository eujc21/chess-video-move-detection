use anyhow::{Context, Result};
use chess_video_moves::geometry::Point;
use chess_video_moves::pipeline::{self, Models, Settings};
use chess_video_moves::yolo::{ComputeUnits, RuntimeOptions};
use chess_video_moves::{default_cache_dir, parse_corners};
use clap::Parser;
use std::io::Write;
use std::path::Path;

/// Extract chess moves from videos of a game.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Input videos.
    #[arg(required = true)]
    videos: Vec<String>,
    /// Output CSV (row_id,output).
    #[arg(short, long, default_value = "result.csv")]
    output: String,
    #[arg(long, default_value = "src/models/board-model.onnx")]
    board_model: String,
    #[arg(long, default_value = "src/models/pieces-model.onnx")]
    pieces_model: String,
    #[arg(long, default_value = "src/models/hand-model.onnx")]
    hand_model: String,
    /// Seconds between analysed frames.
    #[arg(long, default_value_t = 1.0)]
    interval: f64,
    #[arg(long, default_value_t = 0.65)]
    hand_confidence: f32,
    #[arg(long, default_value_t = 0.01)]
    hand_min_coverage: f64,
    /// Frames per second checked for hands (use the video fps to check every frame).
    #[arg(long, default_value_t = 5.0)]
    hand_checks_per_second: f64,
    /// Consecutive agreeing samples required before a position is accepted.
    #[arg(long, default_value_t = 2)]
    stable_samples: usize,
    /// Most plies inferred between two accepted positions.
    #[arg(long, default_value_t = 2)]
    max_plies: usize,
    /// Largest board mismatch tolerated after a matched move.
    #[arg(long, default_value_t = 1.5)]
    max_residual: f32,
    /// Append + / # to moves.
    #[arg(long)]
    check_marks: bool,
    /// Use the original frame-diff move detection instead of legal-move matching.
    #[arg(long)]
    naive: bool,
    /// Fixed board corners in pixels, "x,y x,y x,y x,y" (top-left, top-right,
    /// bottom-right, bottom-left), instead of running the board model.
    #[arg(long, value_parser = parse_corners)]
    board_corners: Option<[Point; 4]>,
    /// ONNX Runtime intra-op threads (0 = default).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Apple Silicon: compute units CoreML may use (gpu = Metal).
    #[arg(long, value_enum, default_value_t = ComputeUnits::All)]
    compute_units: ComputeUnits,
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

fn main() {
    let code = match run() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {e:?}");
            1
        }
    };
    chess_video_moves::exit_process(code)
}

fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    // Models are loaded once and shared by all videos.
    let rt =
        RuntimeOptions { threads: args.threads, compute_units: args.compute_units, cache_dir: default_cache_dir() };
    let mut models = Models::load(&args.board_model, &args.pieces_model, &args.hand_model, args.hand_confidence, &rt)?;
    let settings = Settings {
        interval_seconds: args.interval,
        hand_min_coverage: args.hand_min_coverage,
        hand_checks_per_second: args.hand_checks_per_second,
        stable_samples: args.stable_samples,
        naive: args.naive,
        board_corners: args.board_corners,
        max_plies: args.max_plies,
        max_residual: args.max_residual,
        check_marks: args.check_marks,
    };

    let mut out = std::fs::File::create(&args.output).with_context(|| format!("creating {}", args.output))?;
    writeln!(out, "row_id,output")?;
    for video in &args.videos {
        let moves =
            pipeline::process_video(video, &mut models, &settings).with_context(|| format!("processing {video}"))?;
        let name = Path::new(video).file_name().map_or(video.clone(), |n| n.to_string_lossy().into_owned());
        log::info!("{name}: {moves}");
        writeln!(out, "{},{}", csv_field(&name), csv_field(&moves))?;
    }
    log::info!("Processing complete. Results saved to {}", args.output);
    Ok(())
}
