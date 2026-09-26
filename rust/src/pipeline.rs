//! Per-video / per-stream orchestration: frame sampling, hand gating,
//! board/piece detection and move tracking.

use crate::board::Board;
use crate::geometry::Point;
use crate::notation::{Occupancy, Tracker, TrackerOptions, format_game, mismatch};
use crate::pieces::{LocatedPiece, PieceDetector};
use crate::video::{Frame, FrameReader};
use crate::yolo::{RuntimeOptions, Yolo};
use anyhow::Result;
use shakmaty::{Color, Piece};
use std::collections::VecDeque;

#[derive(Debug, Clone)]
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

impl Default for Settings {
    fn default() -> Self {
        Self {
            interval_seconds: 1.0,
            hand_min_coverage: 0.01,
            hand_checks_per_second: 5.0,
            stable_samples: 2,
            naive: false,
            board_corners: None,
            max_plies: 2,
            max_residual: 1.5,
            check_marks: false,
        }
    }
}

pub struct Models {
    pub board: Yolo,
    pub pieces: PieceDetector,
    pub hands: Yolo,
}

impl Models {
    /// Loads the three ONNX models; `hand_confidence` is the hand model's threshold.
    pub fn load(board: &str, pieces: &str, hands: &str, hand_confidence: f32, rt: &RuntimeOptions) -> Result<Self> {
        let mut models = Self {
            board: Yolo::load(board, rt)?,
            pieces: PieceDetector::new(Yolo::load(pieces, rt)?),
            hands: Yolo::load(hands, rt)?,
        };
        models.hands.conf = hand_confidence;
        Ok(models)
    }
}

/// Snapshot of what the pipeline currently sees, for display.
#[derive(Debug, Clone, Default)]
pub struct View {
    pub board_corners: Option<[Point; 4]>,
    /// Image-space quad of each cell with the piece last observed on it.
    pub cells: Vec<([Point; 4], Option<Piece>)>,
    pub hand: bool,
    pub moves: Vec<String>,
    pub notation: String,
    pub status: String,
}

const STARTING_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Frame-by-frame move extraction for one video or live stream.
///
/// Frames may arrive with gaps (live capture drops frames it cannot keep up
/// with); only frames selected by [`Session::wants`] are used.
pub struct Session {
    settings: Settings,
    pub interval: u64,
    /// Frames between hand checks.
    pub stride: u64,
    /// Frames banned on each side of a hand detection.
    window: u64,
    board: Option<Board>,
    tracker: Option<Tracker>,
    moves: Vec<String>,
    first_move_black: bool,
    /// Recent observations for the stability filter.
    recent: VecDeque<Occupancy>,
    /// Samples waiting for their hand look-ahead window.
    pending: VecDeque<Frame>,
    last_hand: Option<u64>,
    last_observation: Option<Occupancy>,
    hand_now: bool,
    status: String,
}

impl Session {
    pub fn new(fps: f64, settings: Settings) -> Self {
        // Same sampling as the Python code: int(int(fps) * seconds).
        let interval = ((fps as u64) as f64 * settings.interval_seconds).max(1.0) as u64;
        // Hands are checked ~hand_checks_per_second times per second instead of on every frame.
        let stride = (fps / settings.hand_checks_per_second.max(0.001)).floor().max(1.0) as u64;
        Self {
            board: settings.board_corners.and_then(Board::from_corners),
            settings,
            interval,
            stride,
            window: interval,
            tracker: None,
            moves: Vec::new(),
            first_move_black: false,
            recent: VecDeque::new(),
            pending: VecDeque::new(),
            last_hand: None,
            last_observation: None,
            hand_now: false,
            status: "Waiting for frames".into(),
        }
    }

    /// Whether frame `n` (1-based) is used at all.
    pub fn wants(&self, n: u64) -> bool {
        wants_frame(n, self.interval, self.stride)
    }

    /// Forgets the board so it is detected (and its rotation found) again,
    /// e.g. after the camera was moved. The game position is kept.
    pub fn recalibrate(&mut self) {
        self.board = self.settings.board_corners.and_then(Board::from_corners);
        self.pending.clear();
        self.recent.clear();
        self.last_observation = None;
        self.status = "Recalibrating board".into();
    }

    /// Uses fixed board corners (or `None` to go back to the board model) from now on.
    pub fn set_board_corners(&mut self, corners: Option<[Point; 4]>) {
        self.settings.board_corners = corners;
        self.recalibrate();
    }

    pub fn push(&mut self, frame: Frame, models: &mut Models) -> Result<()> {
        let n = frame.number;
        if !self.wants(n) {
            return Ok(());
        }
        let is_sample = n == 1 || n.is_multiple_of(self.interval);

        if self.board.is_none() {
            if !is_sample {
                return Ok(());
            }
            self.board = Board::detect(&mut models.board, &frame)?;
            if self.board.is_none() {
                log::warn!("No chess board detected in frame {n}");
                self.status = "No chess board detected".into();
                return Ok(());
            }
        }

        let board = self.board.as_ref().expect("board detected above");
        self.hand_now = detect_hand(&mut models.hands, board, &frame, &self.settings)?;
        let window = self.window;
        if self.hand_now {
            log::info!(
                "Hand detected in frame {n}; banning frames {}..={}",
                n.saturating_sub(window).max(1),
                n + window
            );
            self.last_hand = Some(n);
            self.pending.retain(|p| p.number + window < n);
            self.status = "Hand over the board".into();
        }

        // Process samples whose look-ahead window has been fully checked.
        while self.pending.front().is_some_and(|p| p.number + window <= n) {
            let p = self.pending.pop_front().expect("front exists");
            self.process_sample(&p, models)?;
        }

        if is_sample {
            if self.last_hand.is_some_and(|h| h + window >= n) {
                log::info!("Skipping frame {n} because it's banned.");
            } else {
                self.pending.push_back(frame);
            }
        }
        Ok(())
    }

    /// Processes samples still waiting for their look-ahead window.
    pub fn finish(&mut self, models: &mut Models) -> Result<()> {
        while let Some(p) = self.pending.pop_front() {
            self.process_sample(&p, models)?;
        }
        Ok(())
    }

    pub fn notation(&self) -> String {
        format_game(&self.moves, self.first_move_black)
    }

    /// The game as PGN, with a FEN header when it did not start from the initial position.
    pub fn pgn(&self) -> String {
        let mut out = String::from("[Event \"Chess video\"]\n[Site \"?\"]\n[Result \"*\"]\n");
        if let Some(fen) = self.tracker.as_ref().and_then(Tracker::start_fen).filter(|f| f != STARTING_FEN) {
            out.push_str(&format!("[SetUp \"1\"]\n[FEN \"{fen}\"]\n"));
        }
        let moves = self.notation();
        out.push('\n');
        out.push_str(&moves);
        out.push_str(if moves.is_empty() { "*\n" } else { " *\n" });
        out
    }

    pub fn view(&self) -> View {
        let mut view = View {
            hand: self.hand_now,
            moves: self.moves.clone(),
            notation: self.notation(),
            status: self.status.clone(),
            ..View::default()
        };
        if let Some(board) = &self.board {
            view.board_corners = Some(board.corners);
            for row in 0..8 {
                for col in 0..8 {
                    let piece = match (board.rotation, &self.last_observation) {
                        (Some(_), Some(obs)) => obs[board.square(row, col) as usize],
                        _ => None,
                    };
                    view.cells.push((board.cells[row][col], piece));
                }
            }
        }
        view
    }

    fn process_sample(&mut self, frame: &Frame, models: &mut Models) -> Result<()> {
        log::info!("Processing frame {}", frame.number);
        let s = &self.settings;
        let Some(board) = self.board.as_mut() else { return Ok(()) };
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
        self.last_observation = Some(obs);

        // Stability filter: require the last `stable_samples` observations to agree
        // (up to piece-type confusions) so a half-finished move is not used.
        self.recent.push_back(obs);
        while self.recent.len() > s.stable_samples.max(1) {
            self.recent.pop_front();
        }
        if self.recent.len() < s.stable_samples.max(1) || self.recent.iter().any(|o| mismatch(o, &obs) >= 1.0) {
            self.status = "Waiting for a stable position".into();
            return Ok(());
        }
        self.status = format!("Tracking ({} pieces)", obs.iter().flatten().count());

        let Some(tracker) = self.tracker.as_mut() else {
            let opts =
                TrackerOptions { max_plies: s.max_plies, max_residual: s.max_residual, check_marks: s.check_marks };
            self.tracker = Some(Tracker::new(&obs, s.naive, opts));
            log::info!("Initialised position from frame {} ({} pieces)", frame.number, obs.iter().flatten().count());
            return Ok(());
        };
        for (mv, color) in tracker.update(&obs) {
            if self.moves.is_empty() && color == Color::Black {
                log::info!("Black move first");
                self.first_move_black = true;
            }
            log::info!("Detected move: {mv}");
            self.moves.push(mv);
        }
        Ok(())
    }
}

/// Whether frame `n` (1-based) is a sample or a hand-check frame.
pub fn wants_frame(n: u64, interval: u64, stride: u64) -> bool {
    n == 1 || n.is_multiple_of(interval) || n.is_multiple_of(stride)
}

fn detect_hand(model: &mut Yolo, board: &Board, frame: &Frame, s: &Settings) -> Result<bool> {
    let pred = model.predict(frame)?;
    Ok(pred.detections.iter().any(|d| board.coverage(d) >= s.hand_min_coverage))
}

/// Processes a whole video file and returns its moves.
pub fn process_video(path: &str, models: &mut Models, s: &Settings) -> Result<String> {
    let info = crate::video::probe(path)?;
    let mut session = Session::new(info.fps, s.clone());
    log::info!(
        "{path}: {:.2} fps, sampling every {} frames, hand check every {} frames",
        info.fps,
        session.interval,
        session.stride
    );
    // Let ffmpeg skip every frame the session does not want.
    let mut reader = FrameReader::open(path, session.interval, session.stride)?;
    while let Some(frame) = reader.next_frame()? {
        session.push(frame, models)?;
    }
    session.finish(models)?;
    Ok(session.notation())
}
