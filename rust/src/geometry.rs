//! Small, dependency-free geometry helpers: points, homographies and convex
//! polygon clipping. These replace the OpenCV / Shapely calls in the Python code.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// A 3x3 projective transform (row-major).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography(pub [[f64; 3]; 3]);

impl Homography {
    /// Solves for the homography mapping `src[i]` to `dst[i]` (4 point pairs),
    /// equivalent to `cv2.getPerspectiveTransform`.
    pub fn from_points(src: &[Point; 4], dst: &[Point; 4]) -> Option<Self> {
        // Unknowns h0..h7 with h8 = 1.
        let mut a = [[0.0f64; 9]; 8];
        for i in 0..4 {
            let (x, y, u, v) = (src[i].x, src[i].y, dst[i].x, dst[i].y);
            a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
            a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
        }
        let h = solve_8x8(a)?;
        Some(Self([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]))
    }

    pub fn apply(&self, p: Point) -> Point {
        let m = &self.0;
        let w = m[2][0] * p.x + m[2][1] * p.y + m[2][2];
        Point::new((m[0][0] * p.x + m[0][1] * p.y + m[0][2]) / w, (m[1][0] * p.x + m[1][1] * p.y + m[1][2]) / w)
    }
}

/// Gaussian elimination with partial pivoting on an augmented 8x9 matrix.
fn solve_8x8(mut a: [[f64; 9]; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let pivot = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        for row in 0..8 {
            if row != col {
                let f = a[row][col] / a[col][col];
                let pivot_row = a[col];
                for (x, p) in a[row].iter_mut().zip(pivot_row).skip(col) {
                    *x -= f * p;
                }
            }
        }
    }
    let mut x = [0.0; 8];
    for i in 0..8 {
        x[i] = a[i][8] / a[i][i];
    }
    Some(x)
}

/// Shoelace area of a simple polygon.
pub fn polygon_area(poly: &[Point]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let (p, q) = (poly[i], poly[(i + 1) % n]);
        s += p.x * q.y - q.x * p.y;
    }
    s.abs() / 2.0
}

/// Sutherland–Hodgman clipping of `subject` against an axis-aligned rectangle.
/// Returns the intersection polygon (exact for convex `subject`).
pub fn clip_to_rect(subject: &[Point], min: Point, max: Point) -> Vec<Point> {
    let mut out: Vec<Point> = subject.to_vec();
    // (axis, bound, keep_greater)
    let edges = [(0, min.x, true), (0, max.x, false), (1, min.y, true), (1, max.y, false)];
    for (axis, bound, keep_greater) in edges {
        if out.is_empty() {
            break;
        }
        let input = std::mem::take(&mut out);
        let coord = |p: &Point| if axis == 0 { p.x } else { p.y };
        let inside = |p: &Point| if keep_greater { coord(p) >= bound } else { coord(p) <= bound };
        for i in 0..input.len() {
            let cur = input[i];
            let prev = input[(i + input.len() - 1) % input.len()];
            let (cin, pin) = (inside(&cur), inside(&prev));
            if cin != pin {
                let t = (bound - coord(&prev)) / (coord(&cur) - coord(&prev));
                out.push(Point::new(prev.x + t * (cur.x - prev.x), prev.y + t * (cur.y - prev.y)));
            }
            if cin {
                out.push(cur);
            }
        }
    }
    out
}

/// Area of intersection between a convex polygon and an axis-aligned rectangle.
pub fn rect_overlap_area(poly: &[Point], min: Point, max: Point) -> f64 {
    polygon_area(&clip_to_rect(poly, min, max))
}

/// Orders four points as top-left, top-right, bottom-right, bottom-left
/// (same heuristic as the Python `order_points`).
pub fn order_points(pts: &[Point]) -> [Point; 4] {
    let by = |f: &dyn Fn(&Point) -> f64, max: bool| {
        *pts.iter()
            .max_by(|a, b| {
                let o = f(a).total_cmp(&f(b));
                if max { o } else { o.reverse() }
            })
            .unwrap()
    };
    let sum = |p: &Point| p.x + p.y;
    let diff = |p: &Point| p.y - p.x;
    [by(&sum, false), by(&diff, false), by(&sum, true), by(&diff, true)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homography_round_trip() {
        let src =
            [Point::new(100.0, 120.0), Point::new(520.0, 90.0), Point::new(600.0, 470.0), Point::new(60.0, 430.0)];
        let dst = [Point::new(0.0, 0.0), Point::new(8.0, 0.0), Point::new(8.0, 8.0), Point::new(0.0, 8.0)];
        let h = Homography::from_points(&src, &dst).unwrap();
        for (s, d) in src.iter().zip(dst.iter()) {
            let p = h.apply(*s);
            assert!((p.x - d.x).abs() < 1e-9 && (p.y - d.y).abs() < 1e-9);
        }
    }

    #[test]
    fn clipping_area() {
        let sq = [Point::new(0.5, 0.5), Point::new(1.5, 0.5), Point::new(1.5, 1.5), Point::new(0.5, 1.5)];
        let a = rect_overlap_area(&sq, Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        assert!((a - 0.25).abs() < 1e-12);
        let none = rect_overlap_area(&sq, Point::new(3.0, 3.0), Point::new(4.0, 4.0));
        assert_eq!(none, 0.0);
    }

    #[test]
    fn orders_corners() {
        let pts = [Point::new(10.0, 90.0), Point::new(90.0, 90.0), Point::new(90.0, 10.0), Point::new(10.0, 10.0)];
        let [tl, tr, br, bl] = order_points(&pts);
        assert_eq!((tl, tr, br, bl), (pts[3], pts[2], pts[1], pts[0]));
    }
}
