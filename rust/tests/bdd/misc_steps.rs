//! Steps for frame_selection.feature and camera_devices.feature.

use crate::ChessWorld;
use chess_video_moves::video::{next_selected, parse_device_list};
use cucumber::gherkin::Step;
use cucumber::{given, then};

#[given(regex = r"^samples every (\d+) frames and hand checks every (\d+) frames$")]
fn frame_plan(w: &mut ChessWorld, interval: u64, stride: u64) {
    w.frame_plan = Some((interval, stride));
}

#[then(expr = "the first decoded frames are {string}")]
fn first_frames(w: &mut ChessWorld, expected: String) {
    let (interval, stride) = w.frame_plan.expect("no frame plan");
    let count = expected.split_whitespace().count();
    let mut last = 0;
    let got: Vec<String> = (0..count)
        .map(|_| {
            last = next_selected(last, interval, stride);
            last.to_string()
        })
        .collect();
    assert_eq!(got.join(" "), expected);
}

#[given("ffmpeg lists these devices:")]
fn ffmpeg_lists(w: &mut ChessWorld, step: &Step) {
    w.devices = parse_device_list(step.docstring.as_deref().expect("step needs a doc string"));
}

#[then("the cameras offered are:")]
fn cameras_offered(w: &mut ChessWorld, step: &Step) {
    let table = step.table.as_ref().expect("step needs a table");
    let expected: Vec<(String, String)> = table.rows[1..].iter().map(|r| (r[0].clone(), r[1].clone())).collect();
    assert_eq!(w.devices, expected);
}
