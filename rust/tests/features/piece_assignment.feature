Feature: Assigning detected pieces to squares
  A piece's box is assigned to the square it overlaps most. Boxes that
  straddle two squares teach the camera's perspective lean, which then
  nudges every piece, and tall pieces most.

  Background:
    Given a top-down board with 100 pixel squares

  Scenario: A box inside one square
    When these pieces are detected:
      | piece      | x1  | y1  | x2  | y2  |
      | white pawn | 410 | 610 | 490 | 690 |
    Then they are assigned to:
      | piece      | row | col |
      | white pawn | 6   | 4   |

  Scenario: A box over two squares goes to the one it covers most
    When these pieces are detected:
      | piece      | x1  | y1  | x2  | y2  |
      | white rook | 400 | 560 | 500 | 680 |
    Then they are assigned to:
      | piece      | row | col |
      | white rook | 6   | 4   |

  Scenario: Without a learned lean, a tall piece stays where its box is
    When these pieces are detected:
      | piece       | x1  | y1  | x2  | y2  |
      | white queen | 210 | 540 | 290 | 599 |
    Then they are assigned to:
      | piece       | row | col |
      | white queen | 5   | 2   |

  Scenario: A learned lean moves a tall piece onto its real square
    When these pieces are detected:
      | piece       | x1  | y1  | x2  | y2  |
      | white rook  | 400 | 560 | 500 | 680 |
      | white queen | 210 | 540 | 290 | 599 |
    Then they are assigned to:
      | piece       | row | col |
      | white rook  | 6   | 4   |
      | white queen | 6   | 2   |

  Scenario: A box off the board is ignored
    When these pieces are detected:
      | piece      | x1  | y1  | x2  | y2  |
      | black king | 900 | 100 | 980 | 180 |
    Then no pieces are assigned
