Feature: Formatting the game as numbered moves

  Scenario Outline: Numbering moves
    Given the moves "<moves>"
    And black <black> the first move
    When the game notation is written
    Then it reads "<notation>"

    Examples:
      | moves        | black     | notation                  |
      | e4 e5 Nf3    | did not make | 1. e4 e5 2. Nf3        |
      | Qh4 g3       | made      | 1... Qh4 2. g3            |
      | Qh4 g3 e5 d4 | made      | 1... Qh4 2. g3 e5 3. d4   |
      |              | did not make | (empty)                |
