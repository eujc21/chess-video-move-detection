//! Board localisation: the image→grid homography, square geometry, hand
//! overlap tests and orientation detection.

use crate::geometry::{Homography, Point, polygon_area, rect_overlap_area};
use crate::video::Frame;
use crate::yolo::Detection;
use shakmaty::{Color, File, Rank, Square};

/// One piece located on the (unrotated) image grid.
#[derive(Debug, Clone, Copy)]
pub struct GridPiece {
    pub row: usize,
    pub col: usize,
    pub color: Color,
}

#[derive(Debug)]
pub struct Board {
    /// TL, TR, BR, BL in image pixels.
    pub corners: [Point; 4],
    area: f64,
    /// Image pixels -> grid coordinates where the board is `[0, 8]²`
    /// (`x` = column, `y` = row).
    to_grid: Homography,
    from_grid: Homography,
    /// Image-space quadrilateral of each cell, indexed `[row][col]`.
    pub cells: [[[Point; 4]; 8]; 8],
    /// Number of clockwise quarter turns of the board in the image; see [`Board::square`].
    pub rotation: Option<u8>,
}

impl Board {
    pub fn from_corners(corners: [Point; 4]) -> Option<Self> {
        let grid = [Point::new(0.0, 0.0), Point::new(8.0, 0.0), Point::new(8.0, 8.0), Point::new(0.0, 8.0)];
        let to_grid = Homography::from_points(&corners, &grid)?;
        let from_grid = Homography::from_points(&grid, &corners)?;
        let mut cells = [[[Point::new(0.0, 0.0); 4]; 8]; 8];
        for (row, cells_row) in cells.iter_mut().enumerate() {
            for (col, cell) in cells_row.iter_mut().enumerate() {
                let (c, r) = (col as f64, row as f64);
                *cell = [(c, r), (c + 1.0, r), (c + 1.0, r + 1.0), (c, r + 1.0)]
                    .map(|(x, y)| from_grid.apply(Point::new(x, y)));
            }
        }
        let area = polygon_area(&corners);
        Some(Self { corners, area, to_grid, from_grid, cells, rotation: None })
    }

    /// Grid cell `(row, col)` containing an image point, if on the board.
    /// O(1) replacement for testing the point against all 64 square polygons.
    pub fn cell_at(&self, p: Point) -> Option<(usize, usize)> {
        let g = self.to_grid.apply(p);
        if !(g.x.is_finite() && g.y.is_finite()) || g.x < 0.0 || g.y < 0.0 || g.x >= 8.0 || g.y >= 8.0 {
            return None;
        }
        Some((g.y as usize, g.x as usize))
    }

    pub fn cell_center(&self, row: usize, col: usize) -> Point {
        // Midpoint of the TL-BR diagonal, as in the Python implementation.
        let [tl, _, br, _] = self.cells[row][col];
        Point::new((tl.x + br.x) / 2.0, (tl.y + br.y) / 2.0)
    }

    /// Fraction of the board area covered by a detection box.
    pub fn coverage(&self, det: &Detection) -> f64 {
        let inter = rect_overlap_area(&self.corners, Point::new(det.x1, det.y1), Point::new(det.x2, det.y2));
        if self.area > 0.0 { inter / self.area } else { 0.0 }
    }

    /// Chess square of a grid cell for the detected rotation.
    ///
    /// Rotation `k` means a1 sits at: 0 bottom-left, 1 top-left, 2 top-right,
    /// 3 bottom-right of the image grid (the board turned `k` quarter turns clockwise).
    pub fn square(&self, row: usize, col: usize) -> Square {
        let (file, rank) = match self.rotation.unwrap_or(0) {
            0 => (col, 7 - row),
            1 => (row, col),
            2 => (7 - col, row),
            _ => (7 - row, 7 - col),
        };
        Square::from_coords(File::new(file as u32), Rank::new(rank as u32))
    }

    /// Mean brightness of the central part of a cell (sampled through the
    /// homography instead of warping the whole image).
    fn cell_brightness(&self, frame: &Frame, row: usize, col: usize) -> f64 {
        const N: usize = 5;
        let mut sum = 0.0;
        for i in 0..N {
            for j in 0..N {
                let g = Point::new(
                    col as f64 + 0.2 + 0.6 * (j as f64 + 0.5) / N as f64,
                    row as f64 + 0.2 + 0.6 * (i as f64 + 0.5) / N as f64,
                );
                let p = self.from_grid.apply(g);
                let x = (p.x.max(0.0) as usize).min(frame.width - 1);
                let y = (p.y.max(0.0) as usize).min(frame.height - 1);
                sum += frame.luma(x, y);
            }
        }
        sum / (N * N) as f64
    }

    /// Determines the board orientation from one frame.
    ///
    /// 1. Square colours: compares the mean brightness of *all* empty cells
    ///    with even vs odd `row + col` (the Python code uses one pair of
    ///    adjacent empty squares). Even cells are light iff a1 is at a
    ///    bottom-left / top-right corner (rotation 0 or 2).
    /// 2. Sides: compares the mean position of white vs black pieces
    ///    (rather than counting pieces in one half).
    pub fn find_rotation(&self, frame: &Frame, pieces: &[GridPiece]) -> u8 {
        let (mut even, mut odd) = ((0.0, 0usize), (0.0, 0usize));
        for row in 0..8 {
            for col in 0..8 {
                if pieces.iter().any(|p| (p.row, p.col) == (row, col)) {
                    continue;
                }
                let b = self.cell_brightness(frame, row, col);
                let acc = if (row + col) % 2 == 0 { &mut even } else { &mut odd };
                acc.0 += b;
                acc.1 += 1;
            }
        }
        let quarter = if even.1 == 0 || odd.1 == 0 {
            log::warn!("No empty squares of both colours; assuming unrotated square colours");
            false
        } else {
            even.0 / (even.1 as f64) < odd.0 / (odd.1 as f64)
        };

        let mean = |color: Color, f: fn(&GridPiece) -> usize| {
            let v: Vec<f64> = pieces.iter().filter(|p| p.color == color).map(|p| f(p) as f64).collect();
            if v.is_empty() { 3.5 } else { v.iter().sum::<f64>() / v.len() as f64 }
        };
        let rotation = if quarter {
            // White is on the left (rotation 1) or right (rotation 3).
            if mean(Color::White, |p| p.col) <= mean(Color::Black, |p| p.col) { 1 } else { 3 }
        } else if mean(Color::White, |p| p.row) >= mean(Color::Black, |p| p.row) {
            0
        } else {
            2
        };
        log::info!("Determined board rotation: {rotation} clockwise quarter turn(s)");
        rotation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_board() -> Board {
        Board::from_corners([
            Point::new(0.0, 0.0),
            Point::new(80.0, 0.0),
            Point::new(80.0, 80.0),
            Point::new(0.0, 80.0),
        ])
        .unwrap()
    }

    #[test]
    fn cells_and_squares() {
        let mut b = square_board();
        assert_eq!(b.cell_at(Point::new(5.0, 75.0)), Some((7, 0)));
        assert_eq!(b.cell_at(Point::new(85.0, 5.0)), None);
        assert_eq!(b.square(7, 0), Square::A1);
        b.rotation = Some(1);
        assert_eq!(b.square(0, 0), Square::A1);
        assert_eq!(b.square(0, 7), Square::A8);
        b.rotation = Some(2);
        assert_eq!(b.square(0, 7), Square::A1);
        b.rotation = Some(3);
        assert_eq!(b.square(7, 7), Square::A1);
        assert_eq!(b.square(0, 7), Square::H1);
        assert_eq!(b.square(7, 0), Square::A8);
    }

    #[test]
    fn detects_rotation() {
        // Synthetic top-down board: light squares where (row + col) is odd
        // means a quarter turn; white pieces on the left -> rotation 1.
        let (w, h) = (80, 80);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = if (y / 10 + x / 10) % 2 == 1 { 220 } else { 40 };
                rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].fill(v);
            }
        }
        let frame = Frame { number: 1, width: w, height: h, rgb };
        let mut pieces = Vec::new();
        for row in 0..8 {
            pieces.push(GridPiece { row, col: 0, color: Color::White });
            pieces.push(GridPiece { row, col: 7, color: Color::Black });
        }
        assert_eq!(square_board().find_rotation(&frame, &pieces), 1);
    }
}
