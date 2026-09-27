//! Steps for board_orientation.feature and piece_assignment.feature.

use crate::ChessWorld;
use crate::camera::{self, grid_pieces, render};
use crate::tracking_steps::{parse_piece, parse_square};
use chess_video_moves::board::Board;
use chess_video_moves::geometry::Point;
use chess_video_moves::notation::occupancy_of;
use chess_video_moves::pieces::class_for_piece;
use chess_video_moves::yolo::Detection;
use cucumber::gherkin::Step;
use cucumber::{given, then, when};
use shakmaty::{Chess, Position};

fn top_down_board() -> Board {
    let corners = [Point::new(0.0, 0.0), Point::new(800.0, 0.0), Point::new(800.0, 800.0), Point::new(0.0, 800.0)];
    Board::from_corners(corners).expect("valid corners")
}

fn board(w: &ChessWorld) -> &Board {
    w.board.as_ref().expect("no board set up")
}

/// Table rows as maps from column name to cell.
fn rows(step: &Step) -> Vec<std::collections::HashMap<String, String>> {
    let table = step.table.as_ref().expect("step needs a table");
    let header = &table.rows[0];
    table.rows[1..].iter().map(|r| header.iter().cloned().zip(r.iter().cloned()).collect()).collect()
}

fn piece_name(name: &str) -> shakmaty::Piece {
    let (color, role) = name.split_once(' ').expect("piece like `white pawn`");
    parse_piece(color, role)
}

#[given(regex = r"^a top-down board rotated (\d) quarter turns$")]
fn rotated_board(w: &mut ChessWorld, turns: u8) {
    let mut b = top_down_board();
    b.rotation = Some(turns);
    w.board = Some(b);
}

#[given("a top-down board with 100 pixel squares")]
fn plain_board(w: &mut ChessWorld) {
    w.board = Some(top_down_board());
}

#[then(regex = r"^the image cell in row (\d) column (\d) is square ([a-h][1-8])$")]
fn cell_is_square(w: &mut ChessWorld, row: usize, col: usize, square: String) {
    assert_eq!(board(w).square(row, col), parse_square(&square));
}

#[given(regex = r"^a camera image of the starting position rotated (\d) quarter turns$")]
fn rotated_image(w: &mut ChessWorld, turns: u8) {
    w.image_turns = Some(turns);
}

#[when("the board rotation is detected")]
fn detect_rotation(w: &mut ChessWorld) {
    let turns = w.image_turns.expect("no image set up");
    let frame = render(1, turns);
    let board = Board::from_corners(camera::board_corners()).expect("valid corners");
    let pieces = grid_pieces(&occupancy_of(Chess::default().board()), turns);
    w.detected_rotation = Some(board.find_rotation(&frame, &pieces));
}

#[then(regex = r"^the detected rotation is (\d)$")]
fn detected_rotation(w: &mut ChessWorld, turns: u8) {
    assert_eq!(w.detected_rotation, Some(turns));
}

#[then(regex = r"^the point \(([\d.]+), ([\d.]+)\) is in row (\d) column (\d)$")]
fn point_in_cell(w: &mut ChessWorld, x: f64, y: f64, row: usize, col: usize) {
    assert_eq!(board(w).cell_at(Point::new(x, y)), Some((row, col)));
}

#[then(regex = r"^the point \(([\d.]+), ([\d.]+)\) is off the board$")]
fn point_off_board(w: &mut ChessWorld, x: f64, y: f64) {
    assert_eq!(board(w).cell_at(Point::new(x, y)), None);
}

#[when("these pieces are detected:")]
fn pieces_detected(w: &mut ChessWorld, step: &Step) {
    let dets: Vec<Detection> = rows(step)
        .iter()
        .map(|r| {
            let n = |k: &str| r[k].parse::<f64>().expect("number");
            Detection {
                x1: n("x1"),
                y1: n("y1"),
                x2: n("x2"),
                y2: n("y2"),
                conf: 0.9,
                class: class_for_piece(piece_name(&r["piece"])),
                mask_coeffs: Vec::new(),
            }
        })
        .collect();
    let board = w.board.take().expect("no board set up");
    w.assigned = w.assigner.assign(&dets, &board);
    w.board = Some(board);
}

#[then("they are assigned to:")]
fn assigned_to(w: &mut ChessWorld, step: &Step) {
    let expected: Vec<(shakmaty::Piece, usize, usize)> = rows(step)
        .iter()
        .map(|r| (piece_name(&r["piece"]), r["row"].parse().unwrap(), r["col"].parse().unwrap()))
        .collect();
    let got: Vec<(shakmaty::Piece, usize, usize)> = w.assigned.iter().map(|p| (p.piece, p.row, p.col)).collect();
    assert_eq!(got, expected);
}

#[then("no pieces are assigned")]
fn none_assigned(w: &mut ChessWorld) {
    assert!(w.assigned.is_empty(), "assigned: {:?}", w.assigned);
}
