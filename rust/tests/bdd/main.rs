//! Gherkin behaviour tests. Features live in `tests/features`; each module
//! here holds the step definitions for one area.
//!
//! Run with `cargo test --test bdd` (add `-- --name <scenario>` to filter).

mod board_steps;
mod camera;
mod misc_steps;
mod session_steps;
mod tracking_steps;

use chess_video_moves::board::Board;
use chess_video_moves::notation::Tracker;
use chess_video_moves::pieces::{LocatedPiece, PieceAssigner};
use cucumber::World;
use shakmaty::{Chess, Color};

#[derive(Debug, Default, World)]
pub struct ChessWorld {
    // Move tracking
    /// The real game, used to produce what the camera sees.
    pub game: Option<Chess>,
    /// Observation the tracker is created from (the first thing the camera sees).
    pub start_observation: Option<chess_video_moves::notation::Occupancy>,
    pub tracker: Option<Tracker>,
    pub check_marks: bool,
    pub recorded: Vec<(String, Color)>,

    // Notation formatting
    pub moves: Vec<String>,
    pub black_first: bool,
    pub written: Option<String>,

    // Board geometry and piece assignment
    pub board: Option<Board>,
    pub image_turns: Option<u8>,
    pub detected_rotation: Option<u8>,
    pub assigner: PieceAssigner,
    pub assigned: Vec<LocatedPiece>,

    // Live session
    pub camera: Option<camera::Camera>,
    pub session: Option<session_steps::SessionResult>,

    // Frame selection and devices
    pub frame_plan: Option<(u64, u64)>,
    pub devices: Vec<(String, String)>,
}

fn main() {
    futures::executor::block_on(ChessWorld::cucumber().fail_on_skipped().run_and_exit("tests/features"));
}
