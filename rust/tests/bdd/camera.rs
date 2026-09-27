//! A scripted camera: renders a top-down board and answers the `Vision`
//! calls from a timeline, standing in for the YOLO models.

use chess_video_moves::board::{Board, GridPiece};
use chess_video_moves::geometry::Point;
use chess_video_moves::notation::{Occupancy, occupancy_of};
use chess_video_moves::pieces::class_for_piece;
use chess_video_moves::pipeline::Vision;
use chess_video_moves::video::Frame;
use chess_video_moves::yolo::Detection;
use shakmaty::{Chess, Position, Square};

/// Image size; the board spans `MARGIN..MARGIN + 8 * CELL` on both axes.
pub const IMAGE: usize = 200;
pub const MARGIN: f64 = 20.0;
pub const CELL: f64 = 20.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Hand {
    OverBoard,
    TouchingCorner,
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub fps: f64,
    /// Quarter turns of the board in the image.
    pub turns: u8,
    /// The real game's starting position.
    pub start: Chess,
    /// `(from, to, position)` time spans in seconds; later entries win.
    pub positions: Vec<(f64, Option<f64>, Occupancy)>,
    pub hands: Vec<(f64, f64, Hand)>,
    pub board_visible_from: f64,
}

pub fn board_corners() -> [Point; 4] {
    let (a, b) = (MARGIN, MARGIN + 8.0 * CELL);
    [Point::new(a, a), Point::new(b, a), Point::new(b, b), Point::new(a, b)]
}

/// Board layout with the given rotation, used to map squares to image cells.
pub fn layout(turns: u8) -> Board {
    let mut board = Board::from_corners(board_corners()).expect("valid corners");
    board.rotation = Some(turns);
    board
}

/// Image cell `(row, col)` showing a square.
pub fn cell_of(layout: &Board, square: Square) -> (usize, usize) {
    (0..8)
        .flat_map(|r| (0..8).map(move |c| (r, c)))
        .find(|&(r, c)| layout.square(r, c) == square)
        .expect("every square has a cell")
}

/// Renders the board's square colours (pieces are only reported through `Vision`).
pub fn render(number: u64, turns: u8) -> Frame {
    let layout = layout(turns);
    let mut rgb = vec![90u8; IMAGE * IMAGE * 3];
    for row in 0..8 {
        for col in 0..8 {
            let sq = layout.square(row, col);
            let light = (u32::from(sq.file()) + u32::from(sq.rank())) % 2 == 1;
            let v = if light { 220 } else { 60 };
            for y in 0..CELL as usize {
                for x in 0..CELL as usize {
                    let (px, py) =
                        (MARGIN as usize + col * CELL as usize + x, MARGIN as usize + row * CELL as usize + y);
                    let i = (py * IMAGE + px) * 3;
                    rgb[i..i + 3].fill(v);
                }
            }
        }
    }
    Frame { number, width: IMAGE, height: IMAGE, rgb }
}

/// Grid pieces as the board would see them, for rotation detection.
pub fn grid_pieces(occ: &Occupancy, turns: u8) -> Vec<GridPiece> {
    let layout = layout(turns);
    (0..64u32)
        .filter_map(|i| {
            let piece = occ[i as usize]?;
            let (row, col) = cell_of(&layout, Square::new(i));
            Some(GridPiece { row, col, color: piece.color })
        })
        .collect()
}

fn detection(x1: f64, y1: f64, x2: f64, y2: f64, class: usize) -> Detection {
    Detection { x1, y1, x2, y2, conf: 0.9, class, mask_coeffs: Vec::new() }
}

impl Camera {
    pub fn new(fps: f64) -> Self {
        Self {
            fps,
            turns: 0,
            start: Chess::default(),
            positions: Vec::new(),
            hands: Vec::new(),
            board_visible_from: 0.0,
        }
    }

    fn time(&self, frame: &Frame) -> f64 {
        frame.number as f64 / self.fps
    }

    fn position_at(&self, t: f64) -> Occupancy {
        self.positions
            .iter()
            .rev()
            .find(|(from, to, _)| t >= *from && to.is_none_or(|to| t < to))
            .map(|p| p.2)
            .unwrap_or_else(|| occupancy_of(self.start.board()))
    }
}

impl Vision for Camera {
    fn find_board(&mut self, frame: &Frame) -> anyhow::Result<Option<[Point; 4]>> {
        Ok((self.time(frame) >= self.board_visible_from).then(board_corners))
    }

    fn detect_hands(&mut self, frame: &Frame) -> anyhow::Result<Vec<Detection>> {
        let t = self.time(frame);
        Ok(self
            .hands
            .iter()
            .filter(|(from, to, _)| t >= *from && t <= *to)
            .map(|(_, _, hand)| match hand {
                Hand::OverBoard => detection(60.0, 60.0, 140.0, 140.0, 0),
                Hand::TouchingCorner => detection(0.0, 0.0, 22.0, 22.0, 0),
            })
            .collect())
    }

    fn detect_pieces(&mut self, frame: &Frame) -> anyhow::Result<Vec<Detection>> {
        let occ = self.position_at(self.time(frame));
        let layout = layout(self.turns);
        Ok((0..64u32)
            .filter_map(|i| {
                let piece = occ[i as usize]?;
                let (row, col) = cell_of(&layout, Square::new(i));
                let (cx, cy) = (MARGIN + (col as f64 + 0.5) * CELL, MARGIN + (row as f64 + 0.5) * CELL);
                Some(detection(cx - 7.0, cy - 7.0, cx + 7.0, cy + 7.0, class_for_piece(piece)))
            })
            .collect())
    }
}
