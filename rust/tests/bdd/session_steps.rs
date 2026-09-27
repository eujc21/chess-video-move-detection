//! Steps for live_session.feature: a `Session` driven by the scripted camera.

use crate::ChessWorld;
use crate::camera::{Camera, Hand, render};
use crate::tracking_steps::{play, position_from_fen};
use chess_video_moves::notation::occupancy_of;
use chess_video_moves::pipeline::{Session, Settings};
use cucumber::{given, then, when};
use shakmaty::Position;

#[derive(Debug)]
pub struct SessionResult {
    pub notation: String,
    pub analysed: Vec<u64>,
    pub pgn: String,
}

fn camera(w: &mut ChessWorld) -> &mut Camera {
    w.camera.as_mut().expect("no camera set up")
}

#[given(regex = r"^a (\d+) fps camera looking down at the starting position$")]
fn camera_setup(w: &mut ChessWorld, fps: f64) {
    w.camera = Some(Camera::new(fps));
    w.session = None;
}

#[given(expr = "the pieces start as {string}")]
fn pieces_start(w: &mut ChessWorld, fen: String) {
    camera(w).start = position_from_fen(&fen);
}

#[given(regex = r"^the camera sees the board rotated (\d) quarter turns$")]
fn camera_rotated(w: &mut ChessWorld, turns: u8) {
    camera(w).turns = turns;
}

#[given(regex = r#"^the position after "([^"]*)" from ([\d.]+) s$"#)]
fn position_from(w: &mut ChessWorld, moves: String, from: f64) {
    add_position(w, &moves, from, None);
}

#[given(regex = r#"^the position after "([^"]*)" from ([\d.]+) s to ([\d.]+) s$"#)]
fn position_between(w: &mut ChessWorld, moves: String, from: f64, to: f64) {
    add_position(w, &moves, from, Some(to));
}

fn add_position(w: &mut ChessWorld, moves: &str, from: f64, to: Option<f64>) {
    let cam = camera(w);
    let mut game = cam.start.clone();
    play(&mut game, moves);
    cam.positions.push((from, to, occupancy_of(game.board())));
}

#[given(regex = r"^a hand is over the board from ([\d.]+) s to ([\d.]+) s$")]
fn hand_over(w: &mut ChessWorld, from: f64, to: f64) {
    camera(w).hands.push((from, to, Hand::OverBoard));
}

#[given(regex = r"^a hand touches the corner of the board from ([\d.]+) s to ([\d.]+) s$")]
fn hand_corner(w: &mut ChessWorld, from: f64, to: f64) {
    camera(w).hands.push((from, to, Hand::TouchingCorner));
}

#[given(regex = r"^the board only becomes visible at ([\d.]+) s$")]
fn board_visible(w: &mut ChessWorld, from: f64) {
    camera(w).board_visible_from = from;
}

#[when(regex = r"^the camera runs for ([\d.]+) s$")]
fn camera_runs(w: &mut ChessWorld, seconds: f64) {
    let mut cam = camera(w).clone();
    let mut session = Session::new(cam.fps, Settings::default());
    let frames = (seconds * cam.fps).round() as u64;
    for n in 1..=frames {
        if session.wants(n) {
            session.push(render(n, cam.turns), &mut cam).expect("session step failed");
        }
    }
    session.finish(&mut cam).expect("session finish failed");
    w.session = Some(SessionResult {
        notation: session.notation(),
        analysed: session.analysed_frames().to_vec(),
        pgn: session.pgn(),
    });
}

#[then(expr = "the analysed frames are {string}")]
fn analysed_frames(w: &mut ChessWorld, expected: String) {
    let got = &w.session.as_ref().expect("camera has not run").analysed;
    let got: Vec<String> = got.iter().map(u64::to_string).collect();
    assert_eq!(got.join(" "), expected);
}

#[then(expr = "the PGN has the FEN header {string}")]
fn pgn_fen(w: &mut ChessWorld, fen: String) {
    let pgn = &w.session.as_ref().expect("camera has not run").pgn;
    assert!(pgn.contains(&format!("[FEN \"{fen}\"]")), "PGN was:\n{pgn}");
}
