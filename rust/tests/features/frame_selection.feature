Feature: Decoding only the frames that are needed
  Only frame 1, the one-per-interval samples and the hand-check frames are
  decoded; everything else is skipped by ffmpeg.

  Scenario Outline: Which frames are decoded
    Given samples every <interval> frames and hand checks every <stride> frames
    Then the first decoded frames are "<frames>"

    Examples:
      | interval | stride | frames                  |
      | 30       | 6      | 1 6 12 18 24 30 36      |
      | 29       | 6      | 1 6 12 18 24 29 30 36   |
      | 1        | 1      | 1 2 3 4 5               |
      | 30       | 10     | 1 10 20 30 40           |
