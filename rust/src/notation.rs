//! Turning a sequence of observed board occupancies into moves.
//!
//! [`LegalTracker`] keeps a real chess position and, for every new
//! observation, searches the legal moves (up to two plies, in case a move was
//! missed while a hand covered the board) for the one whose resulting board
//! best matches what the camera sees. This rejects detector noise that the
//! Python "vacated/occupied tiles" diff would turn into bogus moves, and it
//! gets castling, en passant, promotions, captures and SAN disambiguation
//! right. [`NaiveTracker`] is a straight port of the Python logic, used as a
//! fallback when the first observed position is not a legal chess position.

use shakmaty::san::{San, SanPlus};
use shakmaty::{
    Bitboard, Board, CastlingMode, Chess, Color, FromSetup, Move, Piece, Position, PositionError, Role, Setup, Square,
};
use std::collections::HashSet;

/// Observed piece per square, indexed by `Square as usize`.
pub type Occupancy = [Option<Piece>; 64];

/// Mismatch between two boards. A wrong piece *type* of the right colour
/// counts half: the classifier confuses types far more often than it
/// hallucinates or misses a piece.
pub fn mismatch(a: &Occupancy, b: &Occupancy) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| match (x, y) {
            (None, None) => 0.0,
            (Some(p), Some(q)) if p == q => 0.0,
            (Some(p), Some(q)) if p.color == q.color => 0.5,
            _ => 1.0,
        })
        .sum()
}

fn occupancy_of(board: &Board) -> Occupancy {
    let mut occ = [None; 64];
    for (sq, piece) in board {
        occ[sq as usize] = Some(piece);
    }
    occ
}

pub struct TrackerOptions {
    /// Largest number of plies inferred between two observations.
    pub max_plies: usize,
    /// Largest mismatch allowed between the board after the inferred move(s)
    /// and the observation.
    pub max_residual: f32,
    /// Append `+` / `#` to SAN moves.
    pub check_marks: bool,
}

pub struct LegalTracker {
    /// Candidate positions; starts with both sides to move and collapses to
    /// one as soon as a move is found.
    candidates: Vec<Chess>,
    opts: TrackerOptions,
}

impl LegalTracker {
    /// Builds a tracker from the first observed position, inferring castling
    /// rights from king/rook placement. Returns `None` if the observation is
    /// not a valid position for either side to move.
    pub fn new(first: &Occupancy, opts: TrackerOptions) -> Option<Self> {
        let mut board = Board::empty();
        for (i, p) in first.iter().enumerate() {
            if let Some(p) = p {
                board.set_piece_at(Square::new(i as u32), *p);
            }
        }
        let mut castling = Bitboard::EMPTY;
        for (king, rooks) in [(Square::E1, [Square::A1, Square::H1]), (Square::E8, [Square::A8, Square::H8])] {
            let color = if king == Square::E1 { Color::White } else { Color::Black };
            if board.piece_at(king) == Some(Piece { color, role: Role::King }) {
                for rook in rooks {
                    if board.piece_at(rook) == Some(Piece { color, role: Role::Rook }) {
                        castling.add(rook);
                    }
                }
            }
        }
        let candidates: Vec<Chess> = [Color::White, Color::Black]
            .into_iter()
            .filter_map(|turn| {
                let setup = Setup { board: board.clone(), turn, castling_rights: castling, ..Setup::empty() };
                Chess::from_setup(setup, CastlingMode::Standard)
                    .or_else(PositionError::ignore_too_much_material)
                    .or_else(PositionError::ignore_invalid_castling_rights)
                    .ok()
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        Some(Self { candidates, opts })
    }

    /// Consumes an observation and returns the moves (with their colour)
    /// that explain it, if any.
    pub fn update(&mut self, obs: &Occupancy) -> Vec<(String, Color)> {
        let mut best: Option<(f32, usize, Vec<Move>)> = None; // (cost, candidate, moves)
        for (ci, pos) in self.candidates.iter().enumerate() {
            let stay = mismatch(&occupancy_of(pos.board()), obs);
            if stay == 0.0 {
                return Vec::new();
            }
            if let Some((cost, moves)) = self.search(pos, obs, stay) {
                let better = match &best {
                    None => true,
                    Some((c, _, m)) => cost < *c || (cost == *c && moves.len() < m.len()),
                };
                if better {
                    best = Some((cost, ci, moves));
                }
            }
        }
        let Some((cost, ci, moves)) = best else { return Vec::new() };
        log::debug!("matched {} ply(ies) with residual {cost}", moves.len());

        let mut pos = self.candidates.swap_remove(ci);
        let mut out = Vec::new();
        for m in moves {
            let color = pos.turn();
            let text = if self.opts.check_marks {
                SanPlus::from_move_and_play_unchecked(&mut pos, m).to_string()
            } else {
                let san = San::from_move(&pos, m);
                pos.play_unchecked(m);
                san.to_string()
            };
            out.push((text, color));
        }
        self.candidates = vec![pos];
        out
    }

    /// Best sequence of 1..=max_plies legal moves; only accepted if it explains
    /// the observation better than "nothing happened" and within the residual.
    fn search(&self, pos: &Chess, obs: &Occupancy, stay: f32) -> Option<(f32, Vec<Move>)> {
        let mut best: Option<(f32, Vec<Move>)> = None;
        let mut frontier: Vec<(Chess, Vec<Move>)> = vec![(pos.clone(), Vec::new())];
        for _ in 0..self.opts.max_plies {
            let mut next = Vec::new();
            for (p, line) in &frontier {
                for m in p.legal_moves() {
                    let mut after = p.clone();
                    after.play_unchecked(m);
                    let cost = mismatch(&occupancy_of(after.board()), obs);
                    let mut l = line.clone();
                    l.push(m);
                    // Deeper lines must be strictly better (fewer plies wins ties).
                    if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                        best = Some((cost, l.clone()));
                    }
                    next.push((after, l));
                }
            }
            if best.as_ref().is_some_and(|(c, _)| *c == 0.0) {
                break;
            }
            frontier = next;
        }
        best.filter(|(c, _)| *c < stay && *c <= self.opts.max_residual)
    }
}

/// Port of the Python `NotationGenerator`: diffs consecutive observations.
#[derive(Default)]
pub struct NaiveTracker {
    all_positions: HashSet<Square>,
    previous: Option<Occupancy>,
}

impl NaiveTracker {
    pub fn new(first: &Occupancy) -> Self {
        let all_positions = (0..64).filter(|&i| first[i].is_some()).map(|i| Square::new(i as u32)).collect();
        Self { all_positions, previous: Some(*first) }
    }

    pub fn update(&mut self, obs: &Occupancy) -> Vec<(String, Color)> {
        let prev = self.previous.replace(*obs).unwrap_or(*obs);
        let (mut vacated, mut occupied) = (Vec::new(), Vec::new());
        for i in 0..64 {
            match (prev[i], obs[i]) {
                (Some(_), None) => vacated.push(i),
                (None, Some(_)) => occupied.push(i),
                (Some(p), Some(q)) if p != q => {
                    vacated.push(i);
                    occupied.push(i);
                }
                _ => {}
            }
        }
        let found = if vacated.len() == 1 && occupied.len() == 1 {
            Some((vacated[0], occupied[0])).filter(|(v, o)| v != o)
        } else {
            vacated
                .iter()
                .flat_map(|&v| occupied.iter().map(move |&o| (v, o)))
                .find(|&(v, o)| prev[v].is_some() && prev[v] == obs[o])
        };
        let Some((from, to)) = found else { return Vec::new() };
        let piece = obs[to].expect("occupied square");
        let (from, to) = (Square::new(from as u32), Square::new(to as u32));
        let capture = self.all_positions.contains(&to);
        self.all_positions.remove(&from);
        self.all_positions.insert(to);
        let mut s = String::new();
        if piece.role == Role::Pawn {
            if capture {
                s.push(from.file().char());
                s.push('x');
            }
        } else {
            s.push(piece.role.upper_char());
            if capture {
                s.push('x');
            }
        }
        s.push_str(&to.to_string());
        vec![(s, piece.color)]
    }
}

pub enum Tracker {
    Legal(LegalTracker),
    Naive(NaiveTracker),
}

impl Tracker {
    pub fn new(first: &Occupancy, naive: bool, opts: TrackerOptions) -> Self {
        if !naive {
            if let Some(t) = LegalTracker::new(first, opts) {
                return Tracker::Legal(t);
            }
            log::warn!("First observed position is not a legal chess position; falling back to naive move diffing");
        }
        Tracker::Naive(NaiveTracker::new(first))
    }

    pub fn update(&mut self, obs: &Occupancy) -> Vec<(String, Color)> {
        match self {
            Tracker::Legal(t) => t.update(obs),
            Tracker::Naive(t) => t.update(obs),
        }
    }
}

/// Formats moves as `1. e4 e5 2. Nf3`, or `1... e5 2. Nf3` when black moved first.
pub fn format_game(moves: &[String], first_move_black: bool) -> String {
    let mut parts = Vec::new();
    let mut iter = moves.iter();
    let mut number = 1;
    if first_move_black && let Some(m) = iter.next() {
        parts.push(format!("1... {m}"));
        number = 2;
    }
    let rest: Vec<&String> = iter.collect();
    for pair in rest.chunks(2) {
        let mut s = format!("{number}. {}", pair[0]);
        if let Some(b) = pair.get(1) {
            s.push(' ');
            s.push_str(b);
        }
        parts.push(s);
        number += 1;
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;

    fn occ(fen: &str) -> Occupancy {
        let setup = fen.parse::<Fen>().unwrap().into_setup();
        occupancy_of(&setup.board)
    }

    fn opts() -> TrackerOptions {
        TrackerOptions { max_plies: 2, max_residual: 1.0, check_marks: false }
    }

    #[test]
    fn tracks_moves_and_turn() {
        let start = occ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let mut t = LegalTracker::new(&start, opts()).unwrap();
        let after_e4 = occ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
        assert_eq!(t.update(&after_e4), vec![("e4".to_string(), Color::White)]);
        // Two plies seen at once (a move missed while a hand was over the board).
        let after_nf3 = occ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
        let moves: Vec<String> = t.update(&after_nf3).into_iter().map(|m| m.0).collect();
        assert_eq!(moves, vec!["e5", "Nf3"]);
    }

    #[test]
    fn ignores_single_square_noise() {
        let start = occ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let mut t = LegalTracker::new(&start, opts()).unwrap();
        let mut noisy = start;
        noisy[Square::E4 as usize] = Some(Piece { color: Color::White, role: Role::Pawn });
        assert!(t.update(&noisy).is_empty());
    }

    #[test]
    fn infers_black_to_move_and_castling() {
        let before = occ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        let mut t = LegalTracker::new(&before, opts()).unwrap();
        let after = occ("r4rk1/8/8/8/8/8/8/R3K2R w KQ - 1 2");
        assert_eq!(t.update(&after), vec![("O-O".to_string(), Color::Black)]);
    }

    #[test]
    fn naive_matches_python() {
        let start = occ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let mut t = NaiveTracker::new(&start);
        let after = occ("rnb1kbnr/pppppppp/8/8/7q/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(t.update(&after), vec![("Qh4".to_string(), Color::Black)]);
    }

    #[test]
    fn formats_game() {
        let m: Vec<String> = ["Qh4", "g3", "e5"].iter().map(|s| s.to_string()).collect();
        assert_eq!(format_game(&m, true), "1... Qh4 2. g3 e5");
        assert_eq!(format_game(&m, false), "1. Qh4 g3 2. e5");
        assert_eq!(format_game(&[], false), "");
    }
}
