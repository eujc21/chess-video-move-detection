Feature: Turning observed board positions into moves
  The tracker keeps a legal chess position. Each time the camera reports a
  new stable position it looks for the legal move (or two) that explains it,
  so detector noise is ignored and special moves come out right.

  Background:
    Given a game from the standard starting position

  Scenario: A single pawn move
    When the camera sees the position after "e4"
    Then the recorded moves are "e4"

  Scenario: Two moves seen at once are both recovered
    When the camera sees the position after "e4 e5"
    Then the recorded moves are "e4 e5"

  Scenario: Moves are recorded one at a time through an opening
    When the camera sees each position of "e4 e5 Nf3 Nc6 Bc4 Nf6 O-O"
    Then the recorded moves are "e4 e5 Nf3 Nc6 Bc4 Nf6 O-O"
    And the game notation is "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. O-O"

  Scenario: A piece appearing on one square is noise, not a move
    When the camera sees the current position with a white pawn wrongly on "e4"
    Then no moves are recorded

  Scenario: A move is still found when another piece is not detected
    When the camera sees the position after "Nf3" with the piece on "h2" missing
    Then the recorded moves are "Nf3"

  Scenario: A move is still found when the moved piece is misclassified
    When the camera sees the position after "Nf3" with the piece on "f3" seen as a white bishop
    Then the recorded moves are "Nf3"

  Scenario: An unexplainable change is ignored
    When the camera sees the position after "e4 e5 Nf3"
    Then no moves are recorded

  Scenario: Check marks can be added
    Given check marks are enabled
    When the camera sees each position of "f3 e5 g4 Qh4"
    Then the recorded moves are "f3 e5 g4 Qh4#"

  Scenario Outline: Special moves from a set-up position
    En passant is only possible when the pawn's two-square push was seen.

    Given a game from "<fen>"
    When the camera sees each position of "<moves>"
    Then the recorded moves are "<moves>"

    Examples:
      | fen                                                    | moves     |
      | 4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1                       | O-O-O     |
      | 4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1                      | d5 exd6   |
      | 4k3/1P6/8/8/8/8/8/4K3 w - - 0 1                        | b8=Q      |
      | 4k3/1P6/8/8/8/8/8/4K3 w - - 0 1                        | b8=N      |
      | 4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1                       | Ra2 Ke7   |
      | 7k/8/8/8/8/8/8/N3K2N w - - 0 1                         | Nb3       |
      | 7k/8/8/8/1N3N2/8/8/4K3 w - - 0 1                       | Nbd5      |

  Scenario: Black moving first is detected from the position
    Given a game from "rnbqkbnr/pppp1ppp/8/4p3/5PP1/8/PPPPP2P/RNBQKBNR b KQkq - 0 2"
    When the camera sees the position after "Qh4#"
    Then the recorded moves are "Qh4"
    And the first move was made by black
    And the game notation is "1... Qh4"

  Scenario: An impossible first position falls back to simple diffing
    Given a game from the board "8/8/8/8/8/8/8/Q7" that is not a legal position
    When the camera sees the queen moved from "a1" to "h8"
    Then the recorded moves are "Qh8"
