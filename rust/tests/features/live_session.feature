Feature: Following a game frame by frame
  A session samples one frame per second. Frames within a second of a hand
  over the board are skipped, and a position must be seen on two samples in a
  row before it is used. These scenarios drive a session with a scripted
  10 fps camera looking straight down at the board, instead of real models.

  Background:
    Given a 10 fps camera looking down at the starting position

  Scenario: A move is recorded once it has been stable for two samples
    Given the position after "e4" from 3.5 s
    When the camera runs for 8 s
    Then the game notation is "1. e4"
    And the analysed frames are "1 10 20 30 40 50 60 70 80"

  Scenario: Frames within a second of a hand are not analysed
    Given a hand is over the board from 3 s to 4 s
    When the camera runs for 8 s
    Then the analysed frames are "1 10 60 70 80"

  Scenario: A hand only touching the corner of the board is ignored
    Given a hand touches the corner of the board from 3 s to 4 s
    When the camera runs for 8 s
    Then the analysed frames are "1 10 20 30 40 50 60 70 80"

  Scenario: A move made under the hand is recorded after the hand leaves
    Given a hand is over the board from 3 s to 4 s
    And the position after "e4" from 3.5 s
    When the camera runs for 8 s
    Then the game notation is "1. e4"

  Scenario: Two moves made while a hand stays over the board are both recorded
    Given a hand is over the board from 3 s to 6 s
    And the position after "e4" from 4 s
    And the position after "e4 e5" from 5 s
    When the camera runs for 10 s
    Then the game notation is "1. e4 e5"

  Scenario: A position seen on only one sample is not used
    Given the position after "e4" from 3 s to 3.5 s
    When the camera runs for 8 s
    Then the game notation is ""

  Scenario: Nothing is analysed until the board is found
    Given the board only becomes visible at 2.5 s
    And the position after "d4" from 5 s
    When the camera runs for 8 s
    Then the analysed frames are "30 40 50 60 70 80"
    And the game notation is "1. d4"

  Scenario Outline: The camera can look at the board from any side
    Given the camera sees the board rotated <turns> quarter turns
    And the position after "e4 c5" from 3 s
    When the camera runs for 6 s
    Then the game notation is "1. e4 c5"

    Examples:
      | turns |
      | 0     |
      | 1     |
      | 2     |
      | 3     |

  Scenario: A game that starts mid-way with black to move
    Given the pieces start as "rnbqkbnr/pppp1ppp/8/4p3/5PP1/8/PPPPP2P/RNBQKBNR b KQkq - 0 2"
    And the position after "Qh4#" from 3 s
    When the camera runs for 6 s
    Then the game notation is "1... Qh4"
    And the PGN has the FEN header "rnbqkbnr/pppp1ppp/8/4p3/5PP1/8/PPPPP2P/RNBQKBNR b KQkq - 0 1"
