//! Minimal .npy reader (C-order little-endian float64 only) so the golden tests can
//! compare directly against arrays written by the Python reference.

use std::fs;

pub fn read_npy(path: &str) -> (Vec<usize>, Vec<f64>) {
    let b = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(&b[0..6], b"\x93NUMPY", "{path}: not a npy file");
    let major = b[6];
    let (hlen, hstart) = if major == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10usize)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12usize)
    };
    let header = std::str::from_utf8(&b[hstart..hstart + hlen]).unwrap();
    assert!(header.contains("<f8"), "{path}: expected little-endian float64");
    let ps = header.find("'shape':").expect("no shape") + 8;
    let open = header[ps..].find('(').expect("no (") + ps;
    let close = header[open..].find(')').expect("no )") + open;
    let shape: Vec<usize> = header[open + 1..close]
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect();
    let n: usize = shape.iter().product();
    let off = hstart + hlen;
    let vals = b[off..off + n * 8]
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (shape, vals)
}

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}


/// Returns `true` when the fixture exists.
///
/// The golden fixtures are derived from private image pairs and are deliberately not
/// committed, so a fresh clone runs only the self-contained tests.  Skipping loudly
/// beats failing: a wall of red on a clean checkout is indistinguishable from a real
/// regression.
pub fn have(path: &str) -> bool {
    if std::path::Path::new(path).exists() {
        true
    } else {
        eprintln!("跳过：缺少 {path}（黄金数据由私人照片派生，未纳入版本控制）");
        false
    }
}
