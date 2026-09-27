Feature: Finding cameras
  Cameras are listed from ffmpeg's device listing so they can be picked in the app.

  Scenario: macOS AVFoundation devices
    Given ffmpeg lists these devices:
      """
      [AVFoundation indev @ 0x1] AVFoundation video devices:
      [AVFoundation indev @ 0x1] [0] FaceTime HD Camera
      [AVFoundation indev @ 0x1] [1] Logitech BRIO
      [AVFoundation indev @ 0x1] AVFoundation audio devices:
      [AVFoundation indev @ 0x1] [0] MacBook Pro Microphone
      """
    Then the cameras offered are:
      | device | label                  |
      | 0      | 0: FaceTime HD Camera  |
      | 1      | 1: Logitech BRIO       |

  Scenario: Windows DirectShow devices
    Given ffmpeg lists these devices:
      """
      [dshow @ 0x1] "Integrated Camera" (video)
      [dshow @ 0x1]   Alternative name "@device_pnp_\\?\usb"
      [dshow @ 0x1] "Microphone Array" (audio)
      """
    Then the cameras offered are:
      | device            | label             |
      | Integrated Camera | Integrated Camera |
