//! Read and write Adobe/Resolve `.cube` 3D LUTs.

use std::fs;
use std::io::{BufWriter, Write};

/// Write flat LUT values (`m*3`, R fastest) as a float `.cube`.
pub fn write_cube(path: &str, x: &[f64], n: usize, title: &str) -> std::io::Result<()> {
    let f = fs::File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "TITLE \"{title}\"")?;
    writeln!(w, "LUT_3D_SIZE {n}")?;
    writeln!(w, "DOMAIN_MIN 0 0 0")?;
    writeln!(w, "DOMAIN_MAX 1 1 1")?;
    for i in 0..n * n * n {
        writeln!(w, "{:.7} {:.7} {:.7}", x[i * 3], x[i * 3 + 1], x[i * 3 + 2])?;
    }
    w.flush()
}

pub fn read_cube(path: &str) -> (Vec<f64>, usize) {
    let text = fs::read_to_string(path).expect("read cube");
    let mut n = 0usize;
    let mut vals = Vec::new();
    for line in text.lines() {
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') { continue; }
        let up = s.to_ascii_uppercase();
        if up.starts_with("LUT_3D_SIZE") {
            n = s.split_whitespace().nth(1).unwrap().parse().unwrap();
        } else if up.starts_with("TITLE") || up.starts_with("DOMAIN_") || up.starts_with("LUT_1D") {
            continue;
        } else {
            let p: Vec<f64> = s.split_whitespace().filter_map(|t| t.parse().ok()).collect();
            if p.len() == 3 { vals.extend_from_slice(&p); }
        }
    }
    if n == 0 { n = ((vals.len() / 3) as f64).cbrt().round() as usize; }
    (vals, n)
}
