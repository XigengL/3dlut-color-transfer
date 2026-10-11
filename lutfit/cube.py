"""Read and write Adobe/Resolve ``.cube`` 3D LUTs."""
from __future__ import annotations

import numpy as np


def write_cube(path: str, x: np.ndarray, n: int, *, title: str = "3dlut",
               domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0)) -> None:
    """Write flat LUT values ``x`` (M,3, R fastest) as a float ``.cube``."""
    x = np.asarray(x, dtype=np.float64)
    assert x.shape == (n ** 3, 3), x.shape
    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {n}",
        f"DOMAIN_MIN {domain_min[0]:.10g} {domain_min[1]:.10g} {domain_min[2]:.10g}",
        f"DOMAIN_MAX {domain_max[0]:.10g} {domain_max[1]:.10g} {domain_max[2]:.10g}",
        "",
    ]
    body = "\n".join(f"{r:.7f} {g:.7f} {b:.7f}"
                     for r, g, b in x)
    with open(path, "w") as f:
        f.write("\n".join(lines) + body + "\n")


def read_cube(path: str):
    """Return ``(x, n)`` with ``x`` flat (M,3), R fastest."""
    n = None
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            up = s.upper()
            if up.startswith("LUT_3D_SIZE"):
                n = int(s.split()[1])
            elif up.startswith(("TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_1D_SIZE")):
                continue
            else:
                parts = s.split()
                if len(parts) == 3:
                    vals.append([float(v) for v in parts])
    x = np.asarray(vals, dtype=np.float64)
    if n is None:
        n = int(round(len(x) ** (1.0 / 3.0)))
    assert x.shape == (n ** 3, 3), (x.shape, n)
    return x, n
