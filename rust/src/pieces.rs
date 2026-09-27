//! Assignment of piece detections to board cells.

use crate::board::{Board, GridPiece};
use crate::geometry::{Point, rect_overlap_area};
use crate::yolo::Detection;
use shakmaty::{Color, Piece, Role};

/// Class order of the pieces model.
pub const CLASSES: [(Color, Role); 12] = [
    (Color::Black, Role::Bishop),
    (Color::Black, Role::King),
    (Color::Black, Role::Knight),
    (Color::Black, Role::Pawn),
    (Color::Black, Role::Queen),
    (Color::Black, Role::Rook),
    (Color::White, Role::Bishop),
    (Color::White, Role::King),
    (Color::White, Role::Knight),
    (Color::White, Role::Pawn),
    (Color::White, Role::Queen),
    (Color::White, Role::Rook),
];

#[derive(Debug, Clone, Copy)]
pub struct LocatedPiece {
    pub row: usize,
    pub col: usize,
    pub piece: Piece,
    pub conf: f32,
}

impl LocatedPiece {
    pub fn grid(&self) -> GridPiece {
        GridPiece { row: self.row, col: self.col, color: self.piece.color }
    }
}

/// Piece for a pieces-model class index.
pub fn piece_for_class(class: usize) -> Option<Piece> {
    CLASSES.get(class).map(|&(color, role)| Piece { color, role })
}

/// Pieces-model class index for a piece.
pub fn class_for_piece(piece: Piece) -> usize {
    CLASSES.iter().position(|&(c, r)| c == piece.color && r == piece.role).expect("all pieces have a class")
}

/// Maps piece boxes to board cells, learning the camera's perspective lean.
#[derive(Debug, Default)]
pub struct PieceAssigner {
    /// Running sum / count of the perspective offset vectors. The Python code
    /// keeps an ever-growing list and re-averages it each frame (O(n) per
    /// frame); a running mean gives the same value in O(1).
    offset_sum: (f64, f64),
    offset_count: usize,
}

impl PieceAssigner {
    /// Assigns each detected piece to a cell of the unrotated grid.
    pub fn assign(&mut self, detections: &[Detection], board: &Board) -> Vec<LocatedPiece> {
        let mut assigned = Vec::new();
        for det in detections {
            let Some(piece) = piece_for_class(det.class) else { continue };
            if let Some(cell) = self.max_overlap_cell(board, det) {
                assigned.push((det, cell, piece));
            }
        }

        let mut pieces = Vec::with_capacity(assigned.len());
        for (det, cell, piece) in assigned {
            let (row, col) = if self.offset_count > 0 {
                // Shift the box centre along the learned perspective offset;
                // taller pieces (king/queen) lean further, so use a smaller divisor.
                let factor = match piece.role {
                    Role::Queen | Role::King => 3.0,
                    Role::Rook | Role::Bishop | Role::Knight => 5.0,
                    Role::Pawn => 8.0,
                };
                let n = self.offset_count as f64;
                let p = Point::new(
                    (det.x1 + det.x2) / 2.0 + self.offset_sum.0 / n / factor,
                    (det.y1 + det.y2) / 2.0 + self.offset_sum.1 / n / factor,
                );
                match board.cell_at(p) {
                    Some(c) => c,
                    None => continue,
                }
            } else {
                cell
            };
            pieces.push(LocatedPiece { row, col, piece, conf: det.conf });
        }
        pieces
    }

    /// Cell with the largest box overlap. Also records an offset vector from
    /// the runner-up cell to the best cell when the box straddles two cells.
    fn max_overlap_cell(&mut self, board: &Board, det: &Detection) -> Option<(usize, usize)> {
        let (min, max) = (Point::new(det.x1, det.y1), Point::new(det.x2, det.y2));
        let mut best: Option<(f64, (usize, usize))> = None;
        let mut second: Option<(f64, (usize, usize))> = None;
        for row in 0..8 {
            for col in 0..8 {
                let quad = &board.cells[row][col];
                // Cheap bounding-box rejection before exact clipping.
                if quad.iter().all(|p| p.x < min.x)
                    || quad.iter().all(|p| p.x > max.x)
                    || quad.iter().all(|p| p.y < min.y)
                    || quad.iter().all(|p| p.y > max.y)
                {
                    continue;
                }
                let area = rect_overlap_area(quad, min, max);
                if area <= 0.0 {
                    continue;
                }
                let entry = Some((area, (row, col)));
                if best.is_none_or(|b| area > b.0) {
                    second = best;
                    best = entry;
                } else if second.is_none_or(|s| area > s.0) {
                    second = entry;
                }
            }
        }
        let (best_area, best_cell) = best?;
        if let Some((area, cell)) = second
            && area >= 0.05 * best_area
        {
            let (from, to) = (board.cell_center(cell.0, cell.1), board.cell_center(best_cell.0, best_cell.1));
            self.offset_sum.0 += to.x - from.x;
            self.offset_sum.1 += to.y - from.y;
            self.offset_count += 1;
        }
        Some(best_cell)
    }
}
