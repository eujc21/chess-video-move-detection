//! Desktop app: live webcam (or video file) capture with start/stop, a
//! preview with the detected board and pieces overlaid, and the move list.
//!
//! Threads: the UI thread; a capture thread that reads frames from ffmpeg and
//! updates the preview at full frame rate; and an inference worker. Frames the
//! worker cannot keep up with are dropped, so the analysis stays current
//! instead of falling behind the camera.

use chess_video_moves::default_cache_dir;
use chess_video_moves::geometry::{Point, order_points};
use chess_video_moves::pipeline::{Models, Session, Settings, View, wants_frame};
use chess_video_moves::video::{Frame, FrameReader, Source, list_cameras};
use chess_video_moves::yolo::{ComputeUnits, RuntimeOptions};
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Stroke, TextureHandle, TextureOptions, Vec2};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Live chess move extraction from a webcam or video file.
#[derive(clap::Parser)]
struct Args {
    /// Start with this video file selected instead of a camera.
    #[arg(long)]
    file: Option<String>,
    /// Camera device to select (AVFoundation index/name, /dev/videoN, or DirectShow name).
    #[arg(long)]
    device: Option<String>,
    /// Folder containing board-model.onnx, pieces-model.onnx and hand-model.onnx.
    #[arg(long, default_value = "src/models")]
    models: String,
    /// Fixed board corners "x,y x,y x,y x,y" (TL, TR, BR, BL) instead of the board model.
    #[arg(long, value_parser = chess_video_moves::parse_corners)]
    board_corners: Option<[Point; 4]>,
    /// Start capturing immediately.
    #[arg(long)]
    start: bool,
    /// Graphics backend: wgpu (Metal/Vulkan/DX12) or glow (OpenGL). `auto` uses wgpu on macOS, glow elsewhere.
    #[arg(long, value_enum, default_value_t = RendererChoice::Auto)]
    renderer: RendererChoice,
    /// Open the window, render a few frames and exit 0 (used by CI to check the app starts).
    #[arg(long)]
    smoke_test: bool,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum RendererChoice {
    Auto,
    Glow,
    Wgpu,
}

/// Frames the smoke test renders before closing the window.
const SMOKE_TEST_FRAMES: u32 = 10;

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = <Args as clap::Parser>::parse();
    let mut app = App { model_dir: args.models, ..App::default() };
    app.settings.board_corners = args.board_corners;
    if let Some(device) = args.device {
        app.device = device;
    }
    if let Some(file) = args.file {
        app.kind = SourceKind::File;
        app.file = file;
    }
    let autostart = args.start;
    if args.smoke_test {
        app.smoke_frames_left = Some(SMOKE_TEST_FRAMES);
        // Never let a hung window stall CI.
        std::thread::spawn(|| {
            std::thread::sleep(Duration::from_secs(120));
            eprintln!("smoke test: window did not finish rendering within 120 s");
            std::process::exit(2);
        });
    }
    let renderer = match args.renderer {
        RendererChoice::Glow => eframe::Renderer::Glow,
        RendererChoice::Wgpu => eframe::Renderer::Wgpu,
        // wgpu renders through Metal on macOS; glow (OpenGL) is the most portable elsewhere.
        RendererChoice::Auto if cfg!(target_os = "macos") => eframe::Renderer::Wgpu,
        RendererChoice::Auto => eframe::Renderer::Glow,
    };
    let options = eframe::NativeOptions {
        renderer,
        viewport: egui::ViewportBuilder::default().with_inner_size([1360.0, 820.0]).with_title("Chess Video Moves"),
        ..Default::default()
    };
    eframe::run_native(
        "Chess Video Moves",
        options,
        Box::new(move |cc| {
            if autostart {
                app.start(&cc.egui_ctx);
            }
            Ok(Box::new(app))
        }),
    )
}

#[derive(PartialEq, Clone, Copy)]
enum SourceKind {
    Camera,
    File,
}

/// State shared between the UI and the background threads.
#[derive(Default)]
struct Shared {
    preview: Option<ColorImage>,
    preview_dirty: bool,
    view: View,
    pgn: String,
    status: String,
    error: Option<String>,
}

struct Run {
    stop: Arc<AtomicBool>,
    calibration: Calibration,
    worker: JoinHandle<Option<(ModelKey, Models)>>,
}

/// Pending board-calibration request for the worker: `Some(None)` re-runs the
/// board model, `Some(Some(corners))` switches to manual corners.
type Calibration = Arc<Mutex<Option<Option<[Point; 4]>>>>;

/// Identifies loaded models so they are reused across runs.
#[derive(Clone, PartialEq)]
struct ModelKey {
    dir: String,
    units: ComputeUnits,
}

struct App {
    kind: SourceKind,
    cameras: Vec<(String, String)>,
    device: String,
    width: usize,
    height: usize,
    fps: f64,
    file: String,
    realtime: bool,
    model_dir: String,
    compute_units: ComputeUnits,
    hand_confidence: f32,
    settings: Settings,
    pgn_path: String,

    shared: Arc<Mutex<Shared>>,
    frames_seen: Arc<AtomicU64>,
    frames_analysed: Arc<AtomicU64>,
    run: Option<Run>,
    models: Option<(ModelKey, Models)>,
    texture: Option<TextureHandle>,
    started: Option<Instant>,
    message: String,
    /// `--smoke-test`: frames left to render before closing the window.
    smoke_frames_left: Option<u32>,
    /// Corners clicked so far while picking the board manually.
    picking: Option<Vec<Point>>,
}

impl Default for App {
    fn default() -> Self {
        let cameras = list_cameras();
        let device = cameras
            .first()
            .map(|c| c.0.clone())
            .unwrap_or_else(|| if cfg!(target_os = "macos") { "0".into() } else { "/dev/video0".into() });
        Self {
            kind: SourceKind::Camera,
            cameras,
            device,
            width: 1280,
            height: 720,
            fps: 30.0,
            file: String::new(),
            realtime: true,
            model_dir: "src/models".into(),
            compute_units: ComputeUnits::All,
            hand_confidence: 0.65,
            settings: Settings::default(),
            pgn_path: "game.pgn".into(),
            shared: Arc::default(),
            frames_seen: Arc::default(),
            frames_analysed: Arc::default(),
            run: None,
            models: None,
            texture: None,
            started: None,
            message: String::new(),
            picking: None,
            smoke_frames_left: None,
        }
    }
}

impl App {
    fn running(&self) -> bool {
        self.run.as_ref().is_some_and(|r| !r.worker.is_finished())
    }

    fn start(&mut self, ctx: &egui::Context) {
        let source = match self.kind {
            SourceKind::Camera => {
                Source::Camera { device: self.device.clone(), width: self.width, height: self.height, fps: self.fps }
            }
            SourceKind::File => Source::File { path: self.file.clone(), realtime: self.realtime },
        };
        // Live sources drop frames the worker can't keep up with; offline files never do.
        let live = !matches!(source, Source::File { realtime: false, .. });
        let key = ModelKey { dir: self.model_dir.clone(), units: self.compute_units };
        let models = self.models.take().filter(|(k, _)| *k == key);
        *self.shared.lock().unwrap() = Shared { status: "Starting".into(), ..Shared::default() };
        self.frames_seen.store(0, Ordering::Relaxed);
        self.frames_analysed.store(0, Ordering::Relaxed);
        self.message.clear();

        let job = Job {
            source,
            live,
            key,
            models,
            hand_confidence: self.hand_confidence,
            settings: self.settings.clone(),
            shared: self.shared.clone(),
            frames_seen: self.frames_seen.clone(),
            frames_analysed: self.frames_analysed.clone(),
            stop: Arc::new(AtomicBool::new(false)),
            calibration: Calibration::default(),
            ctx: ctx.clone(),
        };
        let (stop, calibration) = (job.stop.clone(), job.calibration.clone());
        let worker = std::thread::spawn(move || job.run());
        self.run = Some(Run { stop, calibration, worker });
        self.started = Some(Instant::now());
    }

    fn stop(&mut self) {
        if let Some(run) = &self.run {
            run.stop.store(true, Ordering::Relaxed);
        }
    }

    /// Collects a finished run, keeping its models for the next one.
    fn reap(&mut self) {
        if self.run.as_ref().is_some_and(|r| r.worker.is_finished()) {
            let run = self.run.take().unwrap();
            match run.worker.join() {
                Ok(models) => self.models = models,
                Err(_) => self.shared.lock().unwrap().error = Some("worker thread panicked".into()),
            }
            self.started = None;
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let running = self.running();
        ui.heading("Source");
        ui.add_enabled_ui(!running, |ui| {
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.kind, SourceKind::Camera, "Camera");
                ui.radio_value(&mut self.kind, SourceKind::File, "Video file");
            });
            match self.kind {
                SourceKind::Camera => {
                    ui.horizontal(|ui| {
                        let label = self
                            .cameras
                            .iter()
                            .find(|c| c.0 == self.device)
                            .map_or(self.device.clone(), |c| c.1.clone());
                        egui::ComboBox::from_id_salt("camera").selected_text(label).width(170.0).show_ui(ui, |ui| {
                            for (dev, label) in &self.cameras {
                                ui.selectable_value(&mut self.device, dev.clone(), label);
                            }
                        });
                        if ui.button("⟳").on_hover_text("Refresh device list").clicked() {
                            self.cameras = list_cameras();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Device");
                        ui.text_edit_singleline(&mut self.device);
                    });
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.width).range(160..=3840).suffix(" w"));
                        ui.add(egui::DragValue::new(&mut self.height).range(120..=2160).suffix(" h"));
                        ui.add(egui::DragValue::new(&mut self.fps).range(1.0..=120.0).suffix(" fps"));
                    });
                }
                SourceKind::File => {
                    ui.horizontal(|ui| {
                        ui.label("Path");
                        ui.text_edit_singleline(&mut self.file);
                    });
                    ui.checkbox(&mut self.realtime, "Play in real time (like a camera)");
                }
            }

            ui.separator();
            ui.heading("Models");
            ui.horizontal(|ui| {
                ui.label("Folder");
                ui.text_edit_singleline(&mut self.model_dir);
            });
            ui.horizontal(|ui| {
                ui.label("Compute").on_hover_text("CoreML compute units on Apple Silicon (GPU = Metal)");
                egui::ComboBox::from_id_salt("units").selected_text(units_label(self.compute_units)).show_ui(
                    ui,
                    |ui| {
                        for u in [ComputeUnits::All, ComputeUnits::Gpu, ComputeUnits::Ane, ComputeUnits::Cpu] {
                            ui.selectable_value(&mut self.compute_units, u, units_label(u));
                        }
                    },
                );
            });

            ui.separator();
            ui.heading("Detection");
            let s = &mut self.settings;
            egui::Grid::new("settings").num_columns(2).show(ui, |ui| {
                ui.label("Sample every");
                ui.add(egui::DragValue::new(&mut s.interval_seconds).range(0.2..=5.0).speed(0.05).suffix(" s"));
                ui.end_row();
                ui.label("Hand checks");
                ui.add(egui::DragValue::new(&mut s.hand_checks_per_second).range(1.0..=60.0).suffix(" /s"));
                ui.end_row();
                ui.label("Hand confidence");
                ui.add(egui::Slider::new(&mut self.hand_confidence, 0.1..=0.95));
                ui.end_row();
                ui.label("Stable samples");
                ui.add(egui::DragValue::new(&mut s.stable_samples).range(1..=5));
                ui.end_row();
                ui.label("Max plies / gap");
                ui.add(egui::DragValue::new(&mut s.max_plies).range(1..=3));
                ui.end_row();
            });
            ui.checkbox(&mut s.naive, "Naive diffing (no legal-move matching)");
            ui.checkbox(&mut s.check_marks, "Add + / # to moves");
        });

        ui.separator();
        ui.horizontal(|ui| {
            if ui.add_enabled(!running, egui::Button::new("▶ Start").min_size(Vec2::new(90.0, 28.0))).clicked() {
                self.start(ctx);
            }
            if ui.add_enabled(running, egui::Button::new("■ Stop").min_size(Vec2::new(90.0, 28.0))).clicked() {
                self.stop();
            }
        });
        ui.horizontal(|ui| {
            if ui
                .button("Auto-detect board")
                .on_hover_text("Run the board model again, e.g. after moving the camera")
                .clicked()
            {
                self.settings.board_corners = None;
                self.send_calibration(None);
            }
            let label = if self.picking.is_some() { "Cancel picking" } else { "Pick corners…" };
            if ui.button(label).on_hover_text("Click the board's four corners on the preview").clicked() {
                self.picking = if self.picking.is_some() { None } else { Some(Vec::new()) };
            }
        });
        if let Some(points) = &self.picking {
            ui.label(format!("Click corner {} of 4 on the preview", points.len() + 1));
        } else if self.settings.board_corners.is_some() {
            ui.label("Using manually picked corners");
        }

        ui.separator();
        let shared = self.shared.lock().unwrap();
        ui.label(format!("Status: {}", if running { shared.status.as_str() } else { "Stopped" }));
        if let Some(t) = self.started {
            let secs = t.elapsed().as_secs_f64().max(0.001);
            ui.label(format!(
                "Frames: {} read, {} analysed ({:.1}/s)",
                self.frames_seen.load(Ordering::Relaxed),
                self.frames_analysed.load(Ordering::Relaxed),
                self.frames_analysed.load(Ordering::Relaxed) as f64 / secs
            ));
        }
        if running && shared.view.hand {
            ui.colored_label(Color32::from_rgb(230, 80, 60), "✋ Hand over the board");
        }
        if let Some(err) = &shared.error {
            ui.colored_label(Color32::from_rgb(230, 80, 60), err);
        }
    }

    fn send_calibration(&self, corners: Option<[Point; 4]>) {
        if let Some(run) = &self.run {
            *run.calibration.lock().unwrap() = Some(corners);
        }
    }

    fn moves_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Moves");
        let (view, pgn) = {
            let s = self.shared.lock().unwrap();
            (s.view.clone(), s.pgn.clone())
        };
        ui.horizontal(|ui| {
            if ui.button("Copy PGN").clicked() {
                ctx.copy_text(pgn.clone());
                self.message = "PGN copied".into();
            }
            if ui.button("Save PGN").clicked() {
                self.message = match std::fs::write(&self.pgn_path, &pgn) {
                    Ok(()) => format!("Saved {}", self.pgn_path),
                    Err(e) => format!("Save failed: {e}"),
                };
            }
        });
        ui.text_edit_singleline(&mut self.pgn_path);
        if !self.message.is_empty() {
            ui.small(&self.message);
        }
        ui.separator();
        egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
            if view.notation.is_empty() {
                ui.weak("No moves yet");
            } else {
                // One line per move number, e.g. "3. Nf3 Nc6".
                for line in split_move_numbers(&view.notation) {
                    ui.monospace(line);
                }
            }
        });
    }

    fn preview(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let (image, view) = {
            let mut s = self.shared.lock().unwrap();
            let image = if s.preview_dirty { s.preview.clone() } else { None };
            s.preview_dirty = false;
            (image, s.view.clone())
        };
        if let Some(image) = image {
            match &mut self.texture {
                Some(t) if t.size() == image.size => t.set(image, TextureOptions::LINEAR),
                _ => self.texture = Some(ctx.load_texture("preview", image, TextureOptions::LINEAR)),
            }
        }
        let Some(texture) = &self.texture else {
            ui.centered_and_justified(|ui| ui.weak("Press Start to open the camera or video"));
            return;
        };

        // Fit the frame into the panel, keeping its aspect ratio.
        let [w, h] = texture.size();
        let avail = ui.available_size();
        let scale = (avail.x / w as f32).min(avail.y / h as f32);
        let size = Vec2::new(w as f32 * scale, h as f32 * scale);
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
        let painter = ui.painter_at(rect);
        painter.image(texture.id(), rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        let to_screen = |p: Point| Pos2::new(rect.min.x + p.x as f32 * scale, rect.min.y + p.y as f32 * scale);

        // Manual calibration: collect four clicks, then use them as the board corners.
        if let Some(points) = &mut self.picking {
            if response.clicked()
                && let Some(pos) = response.interact_pointer_pos()
            {
                points.push(Point::new(((pos.x - rect.min.x) / scale) as f64, ((pos.y - rect.min.y) / scale) as f64));
            }
            for p in points.iter() {
                painter.circle_filled(to_screen(*p), 6.0, Color32::from_rgb(255, 200, 0));
            }
            if points.len() == 4 {
                let corners = order_points(points);
                self.picking = None;
                self.settings.board_corners = Some(corners);
                self.send_calibration(Some(corners));
            }
        }

        let grid = Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(80, 200, 255, 140));
        for (quad, piece) in &view.cells {
            let pts = quad.map(to_screen);
            painter.line_segment([pts[0], pts[1]], grid);
            painter.line_segment([pts[0], pts[3]], grid);
            if let Some(piece) = piece {
                let c = Pos2::new((pts[0].x + pts[2].x) / 2.0, (pts[0].y + pts[2].y) / 2.0);
                let (fill, text) = match piece.color {
                    shakmaty::Color::White => (Color32::from_rgb(245, 245, 235), Color32::BLACK),
                    shakmaty::Color::Black => (Color32::from_rgb(30, 30, 30), Color32::WHITE),
                };
                painter.circle_filled(c, 10.0, fill);
                painter.text(
                    c,
                    egui::Align2::CENTER_CENTER,
                    piece.role.upper_char(),
                    egui::FontId::monospace(13.0),
                    text,
                );
            }
        }
        if let Some(corners) = view.board_corners {
            let pts: Vec<Pos2> = corners.iter().map(|p| to_screen(*p)).collect();
            let color = if view.hand { Color32::from_rgb(230, 80, 60) } else { Color32::from_rgb(60, 220, 90) };
            painter.add(egui::Shape::closed_line(pts, Stroke::new(3.0_f32, color)));
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.reap();
        egui::SidePanel::left("controls").resizable(false).exact_width(300.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui, ctx));
        });
        egui::SidePanel::right("moves").default_width(220.0).show(ctx, |ui| self.moves_panel(ui, ctx));
        egui::CentralPanel::default().show(ctx, |ui| self.preview(ui, ctx));
        if self.running() {
            // Capture threads request repaints on new frames; this keeps stats ticking.
            ctx.request_repaint_after(Duration::from_millis(250));
        }
        if let Some(left) = &mut self.smoke_frames_left {
            if *left == 0 {
                log::info!("smoke test: rendered {SMOKE_TEST_FRAMES} frames, closing");
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else {
                *left -= 1;
                ctx.request_repaint();
            }
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop();
        if let Some(run) = self.run.take() {
            let _ = run.worker.join();
        }
    }
}

fn units_label(u: ComputeUnits) -> &'static str {
    match u {
        ComputeUnits::All => "Auto (GPU + Neural Engine)",
        ComputeUnits::Gpu => "GPU (Metal)",
        ComputeUnits::Ane => "Neural Engine",
        ComputeUnits::Cpu => "CPU only",
    }
}

/// Splits "1. e4 e5 2. Nf3" into ["1. e4 e5", "2. Nf3"].
fn split_move_numbers(notation: &str) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for tok in notation.split_whitespace() {
        let is_number = tok.ends_with('.') && tok.trim_end_matches('.').chars().all(|c| c.is_ascii_digit());
        match lines.last_mut() {
            Some(line) if !is_number => {
                line.push(' ');
                line.push_str(tok);
            }
            _ => lines.push(tok.to_string()),
        }
    }
    lines
}

/// Everything a run needs, moved into the worker thread.
struct Job {
    source: Source,
    live: bool,
    key: ModelKey,
    models: Option<(ModelKey, Models)>,
    hand_confidence: f32,
    settings: Settings,
    shared: Arc<Mutex<Shared>>,
    frames_seen: Arc<AtomicU64>,
    frames_analysed: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    calibration: Calibration,
    ctx: egui::Context,
}

impl Job {
    /// Runs until the source ends or Stop is pressed; returns the models for reuse.
    fn run(mut self) -> Option<(ModelKey, Models)> {
        let models = match self.models.take() {
            Some(m) => Some(m),
            None => {
                self.set_status("Loading models (first CoreML compile can take a while)");
                let dir = &self.key.dir;
                let rt = RuntimeOptions { threads: 0, compute_units: self.key.units, cache_dir: default_cache_dir() };
                match Models::load(
                    &format!("{dir}/board-model.onnx"),
                    &format!("{dir}/pieces-model.onnx"),
                    &format!("{dir}/hand-model.onnx"),
                    self.hand_confidence,
                    &rt,
                ) {
                    Ok(m) => Some((self.key.clone(), m)),
                    Err(e) => {
                        self.fail(format!("{e:#}"));
                        None
                    }
                }
            }
        };
        let (key, mut models) = models?;
        models.hands.conf = self.hand_confidence;
        if let Err(e) = self.analyse(&mut models) {
            self.fail(format!("{e:#}"));
        }
        self.ctx.request_repaint();
        Some((key, models))
    }

    fn analyse(&self, models: &mut Models) -> anyhow::Result<()> {
        self.set_status("Opening source");
        let mut reader = FrameReader::open_source(&self.source)?;
        let mut session = Session::new(reader.info().fps, self.settings.clone());
        let (interval, stride) = (session.interval, session.stride);

        let (tx, rx): (SyncSender<Frame>, Receiver<Frame>) = sync_channel(1);
        let capture = {
            let (shared, seen, stop, ctx, live) =
                (self.shared.clone(), self.frames_seen.clone(), self.stop.clone(), self.ctx.clone(), self.live);
            std::thread::spawn(move || -> anyhow::Result<()> {
                while !stop.load(Ordering::Relaxed) {
                    let Some(frame) = reader.next_frame()? else { break };
                    seen.fetch_add(1, Ordering::Relaxed);
                    {
                        let mut s = shared.lock().unwrap();
                        s.preview = Some(ColorImage::from_rgb([frame.width, frame.height], &frame.rgb));
                        s.preview_dirty = true;
                    }
                    ctx.request_repaint();
                    if !wants_frame(frame.number, interval, stride) {
                        continue;
                    }
                    if live {
                        // Drop the frame if the worker is still busy; stay real-time.
                        if let Err(TrySendError::Disconnected(_)) = tx.try_send(frame) {
                            break;
                        }
                    } else if tx.send(frame).is_err() {
                        break;
                    }
                }
                Ok(()) // dropping `reader` stops ffmpeg; dropping `tx` ends the worker loop
            })
        };

        self.set_status("Looking for the board");
        for frame in rx {
            if let Some(corners) = self.calibration.lock().unwrap().take() {
                session.set_board_corners(corners);
            }
            session.push(frame, models)?;
            self.frames_analysed.fetch_add(1, Ordering::Relaxed);
            self.publish(&session);
        }
        session.finish(models)?;
        self.publish(&session);
        self.set_status("Finished");
        capture.join().map_err(|_| anyhow::anyhow!("capture thread panicked"))?
    }

    fn publish(&self, session: &Session) {
        let view = session.view();
        let mut s = self.shared.lock().unwrap();
        s.status = view.status.clone();
        s.view = view;
        s.pgn = session.pgn();
        drop(s);
        self.ctx.request_repaint();
    }

    fn set_status(&self, status: &str) {
        self.shared.lock().unwrap().status = status.into();
        self.ctx.request_repaint();
    }

    fn fail(&self, error: String) {
        log::error!("{error}");
        self.shared.lock().unwrap().error = Some(error);
        self.ctx.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use eframe::wgpu::{Backends, Instance};

    /// The app renders through wgpu on macOS; without a native backend compiled in,
    /// wgpu panics at startup ("No wgpu backend feature ... was enabled").
    #[test]
    fn wgpu_has_a_native_backend() {
        let backends = Instance::enabled_backend_features();
        if cfg!(target_os = "macos") {
            assert!(backends.contains(Backends::METAL), "Metal backend missing: {backends:?}");
        } else {
            assert!(backends.intersects(Backends::VULKAN | Backends::GL | Backends::DX12), "no backend: {backends:?}");
        }
    }
}
