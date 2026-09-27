Feature: Working out how the board is turned in the image
  The camera can see the board from any side. Square colours tell a quarter
  turn apart, and which side the white pieces are on tells the rest.

  Scenario Outline: Naming squares for each rotation
    Given a top-down board rotated <turns> quarter turns
    Then the image cell in row <row> column <col> is square <square>

    Examples:
      | turns | row | col | square |
      | 0     | 7   | 0   | a1     |
      | 0     | 0   | 7   | h8     |
      | 1     | 0   | 0   | a1     |
      | 1     | 7   | 0   | h1     |
      | 2     | 0   | 7   | a1     |
      | 2     | 7   | 0   | h8     |
      | 3     | 7   | 7   | a1     |
      | 3     | 0   | 7   | h1     |

  Scenario Outline: Detecting the rotation from the image
    Given a camera image of the starting position rotated <turns> quarter turns
    When the board rotation is detected
    Then the detected rotation is <turns>

    Examples:
      | turns |
      | 0     |
      | 1     |
      | 2     |
      | 3     |

  Scenario: Points are located on the board in constant time
    Given a top-down board with 100 pixel squares
    Then the point (50, 750) is in row 7 column 0
    And the point (799, 0) is in row 0 column 7
    And the point (850, 50) is off the board
