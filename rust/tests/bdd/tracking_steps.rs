//! Steps for move_tracking.feature and game_notation.feature.

use crate::ChessWorld;
use chess_video_moves::notation::{Occupancy, Tracker, TrackerOptions, format_game, occupancy_of};
use cucumber::{given, then, when};
use shakmaty::fen::Fen;
use shakmaty::san::SanPlus;
use shakmaty::{CastlingMode, Chess, Color, Piece, Position, Role, Square};

pub fn parse_square(s: &str) -> Square {
    s.parse().unwrap_or_else(|_| panic!("bad square {s:?}"))
}

pub fn parse_piece(color: &str, role: &str) -> Piece {
    let color = match color {
        "white" => Color::White,
        "black" => Color::Black,
        other => panic!("bad colour {other:?}"),
    };
    let role = match role {
        "pawn" => Role::Pawn,
        "knight" => Role::Knight,
        "bishop" => Role::Bishop,
        "rook" => Role::Rook,
        "queen" => Role::Queen,
        "king" => Role::King,
        other => panic!("bad piece {other:?}"),
    };
    Piece { color, role }
}

/// Plays space-separated SAN moves on a position.
pub fn play(pos: &mut Chess, moves: &str) {
    for san in moves.split_whitespace() {
        let parsed: SanPlus = san.parse().unwrap_or_else(|_| panic!("bad SAN {san:?}"));
        let m = parsed.san.to_move(pos).unwrap_or_else(|e| panic!("illegal move {san}: {e}"));
        pos.play_unchecked(m);
    }
}

pub fn position_from_fen(fen: &str) -> Chess {
    fen.parse::<Fen>()
        .unwrap_or_else(|e| panic!("bad FEN {fen:?}: {e}"))
        .into_position(CastlingMode::Standard)
        .unwrap_or_else(|e| panic!("illegal FEN position {fen:?}: {e}"))
}

impl ChessWorld {
    fn start_game(&mut self, game: Chess) {
        self.start_observation = Some(occupancy_of(game.board()));
        self.game = Some(game);
        self.tracker = None;
        self.recorded.clear();
    }

    fn game_mut(&mut self) -> &mut Chess {
        self.game.as_mut().expect("no game set up")
    }

    fn current_observation(&self) -> Occupancy {
        occupancy_of(self.game.as_ref().expect("no game set up").board())
    }

    /// Feeds one camera observation to the tracker (created on first use).
    fn observe(&mut self, obs: Occupancy) {
        let check_marks = self.check_marks;
        let start = self.start_observation.expect("no starting observation");
        let tracker = self.tracker.get_or_insert_with(|| {
            Tracker::new(&start, false, TrackerOptions { max_plies: 2, max_residual: 1.5, check_marks })
        });
        self.recorded.extend(tracker.update(&obs));
    }
}

#[given("a game from the standard starting position")]
fn standard_game(w: &mut ChessWorld) {
    w.start_game(Chess::default());
}

#[given(expr = "a game from {string}")]
fn game_from_fen(w: &mut ChessWorld, fen: String) {
    w.start_game(position_from_fen(&fen));
}

#[given(expr = "a game from the board {string} that is not a legal position")]
fn illegal_board(w: &mut ChessWorld, board_fen: String) {
    let setup = format!("{board_fen} w - - 0 1").parse::<Fen>().expect("bad board FEN").into_setup();
    w.start_observation = Some(occupancy_of(&setup.board));
    w.game = None;
    w.tracker = None;
}

#[given("check marks are enabled")]
fn check_marks(w: &mut ChessWorld) {
    w.check_marks = true;
}

#[when(expr = "the camera sees the position after {string}")]
fn sees_after(w: &mut ChessWorld, moves: String) {
    play(w.game_mut(), &moves);
    let obs = w.current_observation();
    w.observe(obs);
}

#[when(expr = "the camera sees each position of {string}")]
fn sees_each(w: &mut ChessWorld, moves: String) {
    for m in moves.split_whitespace() {
        play(w.game_mut(), m);
        let obs = w.current_observation();
        w.observe(obs);
    }
}

#[when(expr = "the camera sees the current position with a white pawn wrongly on {string}")]
fn sees_noise(w: &mut ChessWorld, square: String) {
    let mut obs = w.current_observation();
    obs[parse_square(&square) as usize] = Some(Piece { color: Color::White, role: Role::Pawn });
    w.observe(obs);
}

#[when(expr = "the camera sees the position after {string} with the piece on {string} missing")]
fn sees_missing(w: &mut ChessWorld, moves: String, square: String) {
    play(w.game_mut(), &moves);
    let mut obs = w.current_observation();
    obs[parse_square(&square) as usize] = None;
    w.observe(obs);
}

#[when(
    regex = r#"^the camera sees the position after "([^"]*)" with the piece on "([a-h][1-8])" seen as a (white|black) (\w+)$"#
)]
fn sees_misclassified(w: &mut ChessWorld, moves: String, square: String, color: String, role: String) {
    play(w.game_mut(), &moves);
    let mut obs = w.current_observation();
    obs[parse_square(&square) as usize] = Some(parse_piece(&color, &role));
    w.observe(obs);
}

#[when(expr = "the camera sees the queen moved from {string} to {string}")]
fn sees_piece_moved(w: &mut ChessWorld, from: String, to: String) {
    let mut obs = w.start_observation.expect("no starting observation");
    let piece = obs[parse_square(&from) as usize].take();
    obs[parse_square(&to) as usize] = piece;
    w.observe(obs);
}

#[then(expr = "the recorded moves are {string}")]
fn recorded_moves(w: &mut ChessWorld, expected: String) {
    let got: Vec<&str> = w.recorded.iter().map(|(m, _)| m.as_str()).collect();
    assert_eq!(got.join(" "), expected);
}

#[then("no moves are recorded")]
fn no_moves(w: &mut ChessWorld) {
    assert!(w.recorded.is_empty(), "unexpected moves: {:?}", w.recorded);
}

#[then("the first move was made by black")]
fn black_first(w: &mut ChessWorld) {
    assert_eq!(w.recorded.first().map(|m| m.1), Some(Color::Black));
}

#[then(expr = "the game notation is {string}")]
fn game_notation(w: &mut ChessWorld, expected: String) {
    let notation = match &w.session {
        Some(result) => result.notation.clone(),
        None => {
            let moves: Vec<String> = w.recorded.iter().map(|m| m.0.clone()).collect();
            format_game(&moves, w.recorded.first().is_some_and(|m| m.1 == Color::Black))
        }
    };
    assert_eq!(notation, expected);
}

// game_notation.feature

#[given(expr = "the moves {string}")]
fn the_moves(w: &mut ChessWorld, moves: String) {
    w.moves = moves.split_whitespace().map(String::from).collect();
}

#[given(regex = r"^black (made|did not make) the first move$")]
fn black_made_first(w: &mut ChessWorld, made: String) {
    w.black_first = made == "made";
}

#[when("the game notation is written")]
fn write_notation(w: &mut ChessWorld) {
    w.written = Some(format_game(&w.moves, w.black_first));
}

#[then(expr = "it reads {string}")]
fn it_reads(w: &mut ChessWorld, expected: String) {
    let expected = if expected == "(empty)" { String::new() } else { expected };
    assert_eq!(w.written.as_deref(), Some(expected.as_str()));
}
