//! Minimal YOLO (Ultralytics v8/v11 ONNX export) inference: letterboxing,
//! box decoding, NMS and segmentation-mask corner extraction.
//!
//! Export the `.pt` models first, e.g.
//! `yolo export model=src/models/pieces-model.pt format=onnx imgsz=640`.

use crate::geometry::{Point, order_points};
use crate::video::Frame;
use anyhow::{Context, Result, anyhow, bail};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

const MASK_DIM: usize = 32;
const PAD_VALUE: f32 = 114.0 / 255.0;

#[derive(Debug, Clone)]
pub struct Detection {
    /// Box in original image pixels.
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub conf: f32,
    pub class: usize,
    /// Segmentation mask coefficients (empty for detection models).
    pub mask_coeffs: Vec<f32>,
}

impl Detection {
    pub fn corners(&self) -> [Point; 4] {
        [
            Point::new(self.x1, self.y1),
            Point::new(self.x2, self.y1),
            Point::new(self.x2, self.y2),
            Point::new(self.x1, self.y2),
        ]
    }
}

/// Mapping between original image and letterboxed network input.
#[derive(Debug, Clone, Copy)]
pub struct Letterbox {
    pub scale: f64,
    pub pad_x: f64,
    pub pad_y: f64,
}

pub struct Prediction {
    pub detections: Vec<Detection>,
    /// Prototype masks `[32, mh, mw]` for segmentation models.
    protos: Option<(Vec<f32>, usize, usize)>,
    letterbox: Letterbox,
    input_size: usize,
}

pub struct Yolo {
    session: Session,
    input_size: usize,
    pub conf: f32,
    pub iou: f32,
    pub max_det: usize,
}

impl Yolo {
    pub fn load(path: &str, threads: usize) -> Result<Self> {
        let mut builder = Session::builder()
            .map_err(|e| anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{e}"))?;
        if threads > 0 {
            builder = builder.with_intra_threads(threads).map_err(|e| anyhow!("{e}"))?;
        }
        let session = builder
            .commit_from_file(path)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("loading ONNX model {path}"))?;
        let shape = session.inputs()[0]
            .dtype()
            .tensor_shape()
            .map(|s| s.iter().copied().collect::<Vec<i64>>())
            .unwrap_or_default();
        let input_size = match shape.as_slice() {
            [_, _, h, w] if *h > 0 && h == w => *h as usize,
            _ => 640,
        };
        Ok(Self { session, input_size, conf: 0.25, iou: 0.7, max_det: 300 })
    }

    pub fn predict(&mut self, frame: &Frame) -> Result<Prediction> {
        let (input, letterbox) = letterbox(frame, self.input_size);
        let s = self.input_size;
        let tensor = Tensor::from_array(([1usize, 3, s, s], input)).map_err(|e| anyhow!("{e}"))?;
        let outputs = self.session.run(ort::inputs![tensor]).map_err(|e| anyhow!("{e}"))?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let [_, channels, anchors] = shape[..] else { bail!("unexpected YOLO output shape {shape:?}") };
        let (channels, anchors) = (channels as usize, anchors as usize);

        let protos = if outputs.len() > 1 {
            let (pshape, pdata) = outputs[1].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
            let [_, _, mh, mw] = pshape[..] else { bail!("unexpected proto shape {pshape:?}") };
            Some((pdata.to_vec(), mh as usize, mw as usize))
        } else {
            None
        };
        let mask_dim = if protos.is_some() { MASK_DIM } else { 0 };
        let num_classes = channels.checked_sub(4 + mask_dim).context("too few output channels")?;

        let at = |c: usize, i: usize| data[c * anchors + i];
        let mut candidates = Vec::new();
        for i in 0..anchors {
            let (class, conf) = (0..num_classes)
                .map(|c| (c, at(4 + c, i)))
                .fold((0, f32::MIN), |best, cur| if cur.1 > best.1 { cur } else { best });
            if conf < self.conf {
                continue;
            }
            let (cx, cy, w, h) = (at(0, i) as f64, at(1, i) as f64, at(2, i) as f64, at(3, i) as f64);
            let unmap_x = |v: f64| ((v - letterbox.pad_x) / letterbox.scale).clamp(0.0, frame.width as f64);
            let unmap_y = |v: f64| ((v - letterbox.pad_y) / letterbox.scale).clamp(0.0, frame.height as f64);
            candidates.push(Detection {
                x1: unmap_x(cx - w / 2.0),
                y1: unmap_y(cy - h / 2.0),
                x2: unmap_x(cx + w / 2.0),
                y2: unmap_y(cy + h / 2.0),
                conf,
                class,
                mask_coeffs: (0..mask_dim).map(|k| at(4 + num_classes + k, i)).collect(),
            });
        }
        let detections = nms(candidates, self.iou, self.max_det);
        for d in &detections {
            log::trace!(
                "class {} conf {:.3} box ({:.1}, {:.1}, {:.1}, {:.1})",
                d.class,
                d.conf,
                d.x1,
                d.y1,
                d.x2,
                d.y2
            );
        }
        Ok(Prediction { detections, protos, letterbox, input_size: s })
    }
}

impl Prediction {
    /// Corners (TL, TR, BR, BL) of the segmentation mask of `det`, found as the
    /// extreme points of x+y and y-x over mask pixels. For a (convex) board
    /// quadrilateral these are its vertices, so no contour tracing is needed.
    pub fn mask_corners(&self, det: &Detection, width: usize, height: usize) -> Option<[Point; 4]> {
        let (protos, mh, mw) = self.protos.as_ref()?;
        let (mh, mw) = (*mh, *mw);
        // Mask logits on the prototype grid.
        let mut logits = vec![0f32; mh * mw];
        for (k, c) in det.mask_coeffs.iter().enumerate() {
            let plane = &protos[k * mh * mw..(k + 1) * mh * mw];
            for (l, p) in logits.iter_mut().zip(plane) {
                *l += c * p;
            }
        }
        let lb = self.letterbox;
        let (sx, sy) = (mw as f64 / self.input_size as f64, mh as f64 / self.input_size as f64);
        let sample = |x: f64, y: f64| -> f32 {
            // Bilinear interpolation with half-pixel centres.
            let (fx, fy) = ((x * sx - 0.5).clamp(0.0, (mw - 1) as f64), (y * sy - 0.5).clamp(0.0, (mh - 1) as f64));
            let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(mw - 1), (y0 + 1).min(mh - 1));
            let (tx, ty) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
            let v = |xx: usize, yy: usize| logits[yy * mw + xx];
            let top = v(x0, y0) * (1.0 - tx) + v(x1, y0) * tx;
            let bottom = v(x0, y1) * (1.0 - tx) + v(x1, y1) * tx;
            top * (1.0 - ty) + bottom * ty
        };

        let mut extremes: Option<[Point; 4]> = None;
        let (xa, xb) = (det.x1.floor().max(0.0) as usize, (det.x2.ceil() as usize).min(width));
        let (ya, yb) = (det.y1.floor().max(0.0) as usize, (det.y2.ceil() as usize).min(height));
        for y in ya..yb {
            for x in xa..xb {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                // sigmoid(logit) > 0.5  <=>  logit > 0
                if sample(px * lb.scale + lb.pad_x, py * lb.scale + lb.pad_y) <= 0.0 {
                    continue;
                }
                let p = Point::new(px, py);
                let e = extremes.get_or_insert([p; 4]);
                if p.x + p.y < e[0].x + e[0].y {
                    e[0] = p;
                }
                if p.y - p.x < e[1].y - e[1].x {
                    e[1] = p;
                }
                if p.x + p.y > e[2].x + e[2].y {
                    e[2] = p;
                }
                if p.y - p.x > e[3].y - e[3].x {
                    e[3] = p;
                }
            }
        }
        extremes.map(|e| order_points(&e))
    }
}

/// Resizes (bilinear, aspect-preserving) and pads the frame into a
/// normalised CHW float tensor, like Ultralytics' `LetterBox`.
fn letterbox(frame: &Frame, size: usize) -> (Vec<f32>, Letterbox) {
    let (w, h) = (frame.width as f64, frame.height as f64);
    let scale = (size as f64 / w).min(size as f64 / h);
    let (nw, nh) = ((w * scale).round() as usize, (h * scale).round() as usize);
    let pad_x = ((size - nw) as f64 / 2.0 - 0.1).round().max(0.0);
    let pad_y = ((size - nh) as f64 / 2.0 - 0.1).round().max(0.0);
    let (px, py) = (pad_x as usize, pad_y as usize);

    let plane = size * size;
    let mut out = vec![PAD_VALUE; 3 * plane];
    let (rx, ry) = (w / nw as f64, h / nh as f64);
    let max_x = frame.width - 1;
    let max_y = frame.height - 1;
    for oy in 0..nh {
        let fy = ((oy as f64 + 0.5) * ry - 0.5).clamp(0.0, max_y as f64);
        let (y0, ty) = (fy.floor() as usize, fy.fract() as f32);
        let y1 = (y0 + 1).min(max_y);
        for ox in 0..nw {
            let fx = ((ox as f64 + 0.5) * rx - 0.5).clamp(0.0, max_x as f64);
            let (x0, tx) = (fx.floor() as usize, fx.fract() as f32);
            let x1 = (x0 + 1).min(max_x);
            let (a, b, c, d) = (frame.pixel(x0, y0), frame.pixel(x1, y0), frame.pixel(x0, y1), frame.pixel(x1, y1));
            let idx = (oy + py) * size + ox + px;
            for ch in 0..3 {
                let top = a[ch] as f32 * (1.0 - tx) + b[ch] as f32 * tx;
                let bottom = c[ch] as f32 * (1.0 - tx) + d[ch] as f32 * tx;
                out[ch * plane + idx] = (top * (1.0 - ty) + bottom * ty) / 255.0;
            }
        }
    }
    (out, Letterbox { scale, pad_x, pad_y })
}

fn iou(a: &Detection, b: &Detection) -> f64 {
    let iw = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0.0);
    let ih = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0.0);
    let inter = iw * ih;
    let union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Greedy per-class non-maximum suppression, sorted by confidence.
fn nms(mut dets: Vec<Detection>, iou_thresh: f32, max_det: usize) -> Vec<Detection> {
    dets.sort_by(|a, b| b.conf.total_cmp(&a.conf));
    let mut keep: Vec<Detection> = Vec::new();
    for d in dets {
        if keep.len() >= max_det {
            break;
        }
        if keep.iter().all(|k| k.class != d.class || iou(k, &d) <= iou_thresh as f64) {
            keep.push(d);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f64, y1: f64, x2: f64, y2: f64, conf: f32, class: usize) -> Detection {
        Detection { x1, y1, x2, y2, conf, class, mask_coeffs: vec![] }
    }

    #[test]
    fn nms_suppresses_same_class_only() {
        let kept = nms(
            vec![
                det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
                det(1.0, 1.0, 10.0, 10.0, 0.8, 0),
                det(1.0, 1.0, 10.0, 10.0, 0.7, 1),
            ],
            0.7,
            300,
        );
        assert_eq!(kept.len(), 2);
        assert_eq!((kept[0].class, kept[1].class), (0, 1));
    }

    #[test]
    fn letterbox_pads_to_square() {
        let frame = Frame { number: 1, width: 4, height: 2, rgb: vec![255; 4 * 2 * 3] };
        let (t, lb) = letterbox(&frame, 8);
        assert_eq!(t.len(), 3 * 64);
        assert_eq!((lb.scale, lb.pad_x, lb.pad_y), (2.0, 0.0, 2.0));
        assert_eq!(t[0], PAD_VALUE);
        assert_eq!(t[2 * 8], 1.0);
    }
}
