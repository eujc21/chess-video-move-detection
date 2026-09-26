//! Per-video orchestration: frame sampling, hand gating, board/piece
//! detection and move tracking.

use crate::board::Board;
use crate::geometry::Point;
use crate::notation::{Occupancy, Tracker, TrackerOptions, format_game, mismatch};
use crate::pieces::{LocatedPiece, PieceDetector};
use crate::video::{Frame, FrameReader};
use crate::yolo::Yolo;
use anyhow::Result;
use shakmaty::Color;
use std::collections::VecDeque;

pub struct Settings {
    pub interval_seconds: f64,
    /// Minimum fraction of the board a hand box must cover to count.
    pub hand_min_coverage: f64,
    /// How many frames per second are checked for hands.
    pub hand_checks_per_second: f64,
    /// Consecutive (near-)identical samples required before a position is used.
    pub stable_samples: usize,
    pub naive: bool,
    /// Fixed board corners (TL, TR, BR, BL) instead of running the board model.
    pub board_corners: Option<[Point; 4]>,
    pub max_plies: usize,
    pub max_residual: f32,
    pub check_marks: bool,
}

pub struct Models {
    pub board: Yolo,
    pub pieces: PieceDetector,
    pub hands: Yolo,
}

struct VideoState {
    board: Option<Board>,
    tracker: Option<Tracker>,
    moves: Vec<String>,
    first_move_black: bool,
    /// Recent observations for the stability filter.
    recent: VecDeque<Occupancy>,
}

pub fn process_video(path: &str, models: &mut Models, s: &Settings) -> Result<String> {
    let info = crate::video::probe(path)?;
    // Same sampling as the Python code: int(int(fps) * seconds).
    let interval = ((info.fps as u64) as f64 * s.interval_seconds).max(1.0) as u64;
    let window = interval; // frames banned on each side of a hand detection
    // Only sample frames and every `stride`-th frame are decoded; hands are
    // checked ~hand_checks_per_second times per second instead of on every frame.
    let stride = (info.fps / s.hand_checks_per_second.max(0.001)).floor().max(1.0) as u64;
    log::info!("{path}: {:.2} fps, sampling every {interval} frames, hand check every {stride} frames", info.fps);

    let mut reader = FrameReader::open(path, interval, stride)?;
    let mut st = VideoState {
        board: s.board_corners.and_then(Board::from_corners),
        tracker: None,
        moves: Vec::new(),
        first_move_black: false,
        recent: VecDeque::new(),
    };
    let mut pending: VecDeque<Frame> = VecDeque::new();
    let mut last_hand: Option<u64> = None;

    while let Some(frame) = reader.next_frame()? {
        let n = frame.number;
        let is_sample = n == 1 || n % interval == 0;

        if st.board.is_none() && is_sample {
            st.board = Board::detect(&mut models.board, &frame)?;
            if st.board.is_none() {
                log::warn!("No chess board detected in frame {n}");
                continue;
            }
        }

        let hand = match &st.board {
            Some(board) => detect_hand(&mut models.hands, board, &frame, s)?,
            None => false,
        };
        if hand {
            log::info!(
                "Hand detected in frame {n}; banning frames {}..={}",
                n.saturating_sub(window).max(1),
                n + window
            );
            last_hand = Some(n);
            pending.retain(|p| p.number + window < n);
        }

        // Process samples whose look-ahead window has been fully checked.
        while pending.front().is_some_and(|p| p.number + window <= n) {
            let p = pending.pop_front().unwrap();
            process_sample(&p, models, &mut st, s)?;
        }

        if is_sample && st.board.is_some() {
            if last_hand.is_some_and(|h| h + window >= n) {
                log::info!("Skipping frame {n} because it's banned.");
            } else {
                pending.push_back(frame);
            }
        }
    }
    for p in pending {
        process_sample(&p, models, &mut st, s)?;
    }
    Ok(format_game(&st.moves, st.first_move_black))
}

fn detect_hand(model: &mut Yolo, board: &Board, frame: &Frame, s: &Settings) -> Result<bool> {
    let pred = model.predict(frame)?;
    Ok(pred.detections.iter().any(|d| board.coverage(d) >= s.hand_min_coverage))
}

fn process_sample(frame: &Frame, models: &mut Models, st: &mut VideoState, s: &Settings) -> Result<()> {
    log::info!("Processing frame {}", frame.number);
    let Some(board) = st.board.as_mut() else { return Ok(()) };
    let pieces = models.pieces.detect(frame, board)?;
    if board.rotation.is_none() {
        // The rotation only changes the cell -> square mapping, so the same
        // detections are reused (the Python code runs the model twice).
        let grid: Vec<_> = pieces.iter().map(LocatedPiece::grid).collect();
        board.rotation = Some(board.find_rotation(frame, &grid));
    }

    let mut obs: Occupancy = [None; 64];
    let mut conf = [0f32; 64];
    for p in &pieces {
        let i = board.square(p.row, p.col) as usize;
        // Two detections on one square: keep the more confident one.
        if obs[i].is_none() || p.conf > conf[i] {
            obs[i] = Some(p.piece);
            conf[i] = p.conf;
        }
    }

    // Stability filter: require the last `stable_samples` observations to agree
    // (up to piece-type confusions) so a half-finished move is not used.
    st.recent.push_back(obs);
    while st.recent.len() > s.stable_samples.max(1) {
        st.recent.pop_front();
    }
    if st.recent.len() < s.stable_samples.max(1) || st.recent.iter().any(|o| mismatch(o, &obs) >= 1.0) {
        return Ok(());
    }

    let Some(tracker) = st.tracker.as_mut() else {
        let opts = TrackerOptions { max_plies: s.max_plies, max_residual: s.max_residual, check_marks: s.check_marks };
        st.tracker = Some(Tracker::new(&obs, s.naive, opts));
        log::info!("Initialised position from frame {} ({} pieces)", frame.number, obs.iter().flatten().count());
        return Ok(());
    };
    for (mv, color) in tracker.update(&obs) {
        if st.moves.is_empty() && color == Color::Black {
            log::info!("Black move first");
            st.first_move_black = true;
        }
        log::info!("Detected move: {mv}");
        st.moves.push(mv);
    }
    Ok(())
}
