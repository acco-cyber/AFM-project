#!/usr/bin/env python3
"""
Portion 1 - Dataset audit, preprocessing and leakage control (CPU only)
=======================================================================

DNA-origami HS-AFM dataset, Zenodo record 14275812 (Pothineni, Barner, Keller).
Data format verified against the real archives: every frame is one JPK ``.jpk``
file (a TIFF container). Page 0 holds acquisition metadata; pages 1-6 hold
channels (amplitude, phase, height - each as trace and retrace). Integer data is
converted to physical units with the linear scaling stored in the JPK tags.

Steps (run all, or any subset with --steps):
  verify      md5 checksums of the zip archives against the Zenodo record
  index       list every frame, read its header, build the metadata table
  preprocess  read the height channel (nm), flatten with a fixed protocol,
              compute QC metrics, write per-session stacks (.npy)
  qc          flag problem frames (non-finite, saturated, line artifacts,
              trace/retrace mismatch, noise, duplicates, time gaps)
  splits      experiment-level split manifests + leakage checks
  audit       nearest-neighbour similarity audit between train and test frames
  figures     QC / dataset figures (PNG 300 dpi + PDF) for the paper
  report      summary.json + summary.md

Usage:
  python portion1_pipeline.py --data /path/to/folder_with_zips --out /path/to/output
  python portion1_pipeline.py --data ... --out ... --steps qc splits audit figures report

The input folder may contain the original zip archives (2nM.zip ... 12nM.zip,
overview4x4um.zip), extracted folders with the same names, or both.
Nothing in the input folder is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_VERSION = "p1-1.0.0"

# md5 checksums published on the Zenodo record
ZENODO_MD5 = {
    "2nM.zip": "85067e45aacfb7a35cfba8e2e0753399",
    "4nM.zip": "5f6fc4524e74407d0dd7f6063dbc35f0",
    "6nM.zip": "4430c2042613e3914b08825bccd3c73d",
    "8nM.zip": "e0e4c651fdcb8941f9fd2e38511ffcad",
    "10nM.zip": "e156960f1a0214db2e3213ae5725ce4f",
    "12nM.zip": "603cebe811f74911c7b7ea0777e7bea3",
    "overview4x4um.zip": "34adcf2d75988f14c38c65174908955b",
}
CONCENTRATIONS = [2, 4, 6, 8, 10, 12]
PUBLISHED_TOTAL_ANALYZED = 4267  # images analyzed in the paper (reference only)

log = logging.getLogger("portion1")


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    data: Path
    out: Path
    workers: int = 2
    max_frames: int = 0            # 0 = all; >0 = first N frames per session (quick test)
    feature_thr_nm: float = 1.0    # height above background treated as "feature" for masks
    save_raw: bool = True          # also save plane-levelled (unflattened) stacks
    skip_overview: bool = False
    seed: int = 0


# ----------------------------------------------------------------------------
# Source discovery: zips and/or extracted folders
# ----------------------------------------------------------------------------
CONC_RE = re.compile(r"(?<!\d)(\d{1,2})\s*nM", re.IGNORECASE)
TS_RE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})-(\d{2})\.(\d{2})\.(\d{2})\.(\d{3})")


def parse_conc(text: str):
    m = CONC_RE.search(text)
    return int(m.group(1)) if m else None


def parse_ts(name: str):
    m = TS_RE.search(os.path.basename(name))
    if not m:
        return None
    y, mo, d, h, mi, s, ms = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s, ms * 1000)


def discover_sources(data: Path):
    """Return a list of (source_kind, source_path, member, is_overview, conc_hint).

    source_kind is 'zip' or 'file'. member is the path inside the zip (or None).
    """
    items = []
    seen_zip_stems = set()
    for zp in sorted(data.glob("*.zip")):
        stem = zp.stem
        is_ov = "overview" in stem.lower()
        try:
            with zipfile.ZipFile(zp) as z:
                for info in z.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".jpk"):
                        continue
                    conc = parse_conc(info.filename) if is_ov else parse_conc(stem)
                    items.append(("zip", str(zp), info.filename, is_ov, conc))
            seen_zip_stems.add(stem.lower())
        except zipfile.BadZipFile:
            log.error("Corrupt or incomplete zip, skipped: %s", zp)
    for d in sorted(p for p in data.iterdir() if p.is_dir()):
        if d.name.lower() in seen_zip_stems:
            continue  # same content already read from the zip
        is_ov = "overview" in d.name.lower()
        dconc = parse_conc(d.name)
        if dconc is None and not is_ov:
            continue
        for f in sorted(d.rglob("*.jpk")):
            rel = f.relative_to(d).as_posix()
            conc = parse_conc(rel) if is_ov else dconc
            items.append(("file", str(f), None, is_ov, conc))
    return items


def read_bytes(kind: str, path: str, member):
    if kind == "zip":
        with zipfile.ZipFile(path) as z:
            return z.read(member)
    with open(path, "rb") as fh:
        return fh.read()


# ----------------------------------------------------------------------------
# JPK reader (verified on real files from this dataset)
# ----------------------------------------------------------------------------
TAG_START = 32771
TAG_END = 32774
TAG_X0, TAG_Y0, TAG_ULEN, TAG_VLEN, TAG_ANGLE = 32832, 32833, 32834, 32835, 32836
TAG_ILEN, TAG_JLEN = 32838, 32839
TAG_CH_NAME, TAG_CH_RETRACE = 32848, 32849
TAG_NSLOTS, TAG_DEFAULT_SLOT = 32896, 32897
SLOT_BASE, SLOT_STRIDE = 32912, 48


def _tag(page, code, default=None):
    t = page.tags.get(code)
    return default if t is None else t.value


def _slot_scaling(page, slot_name):
    """Return (multiplier, offset, unit) mapping raw integers to slot_name."""
    n = int(_tag(page, TAG_NSLOTS, 0) or 0)
    for k in range(n):
        b = SLOT_BASE + SLOT_STRIDE * k
        if _tag(page, b) == slot_name:
            mult = _tag(page, b + 20)
            off = _tag(page, b + 21)
            unit = _tag(page, b + 18)
            if mult is None:
                return 1.0, 0.0, None
            return float(mult), float(off or 0.0), unit
    raise KeyError(f"slot {slot_name!r} not found")


def jpk_header(buf: bytes) -> dict:
    import tifffile
    with tifffile.TiffFile(io.BytesIO(buf)) as t:
        p0 = t.pages[0]
        hdr = {
            "start_str": _tag(p0, TAG_START),
            "end_str": _tag(p0, TAG_END),
            "x0_m": _tag(p0, TAG_X0),
            "y0_m": _tag(p0, TAG_Y0),
            "scan_u_m": _tag(p0, TAG_ULEN),
            "scan_v_m": _tag(p0, TAG_VLEN),
            "angle": _tag(p0, TAG_ANGLE),
            "nx": _tag(p0, TAG_ILEN),
            "ny": _tag(p0, TAG_JLEN),
        }
        chans = []
        for i, p in enumerate(t.pages[1:], 1):
            chans.append((i, _tag(p, TAG_CH_NAME), _tag(p, TAG_CH_RETRACE),
                          p.shape))
        hdr["channels"] = chans
    return hdr


def jpk_height(buf: bytes, retrace: bool = False):
    """Return calibrated height in nm (float32), saturation fraction and the fraction
    of rows that were never scanned (set to NaN)."""
    import tifffile
    with tifffile.TiffFile(io.BytesIO(buf)) as t:
        page = None
        for p in t.pages[1:]:
            if str(_tag(p, TAG_CH_NAME)).lower() == "height" and \
                    int(_tag(p, TAG_CH_RETRACE, 0) or 0) == int(retrace):
                page = p
                break
        if page is None:
            raise KeyError("height channel not found")
        raw = page.asarray()
        # rows that were never scanned (frame stopped early) are constant -> invalid
        dead_rows = np.ptp(raw, axis=1) == 0
        slot = _tag(page, TAG_DEFAULT_SLOT) or "calibrated"
        try:
            mult, off, unit = _slot_scaling(page, slot)
        except KeyError:
            mult, off, unit = _slot_scaling(page, "nominal")
        if unit not in (None, "m"):
            raise ValueError(f"unexpected height unit {unit!r}")
        z_nm = (raw.astype(np.float64) * mult + off) * 1e9
        valid = raw[~dead_rows] if (~dead_rows).any() else raw
        rmin, rmax = valid.min(), valid.max()
        sat = float(max((valid == rmin).mean(), (valid == rmax).mean()))
        z_nm[dead_rows, :] = np.nan
    return z_nm.astype(np.float32), sat, float(dead_rows.mean())


# ----------------------------------------------------------------------------
# Flattening protocol (fixed, versioned)
# ----------------------------------------------------------------------------
def _poly_terms(xx, yy, order):
    terms = [np.ones(np.broadcast(xx, yy).shape)]
    if order >= 1:
        terms += [xx + 0 * yy, yy + 0 * xx]
    if order >= 2:
        terms += [xx * xx + 0 * yy, xx * yy, yy * yy + 0 * xx]
    return terms


def _filled(img):
    img = np.asarray(img, dtype=np.float64)
    if np.isfinite(img).all():
        return img
    fill = np.nanmedian(img) if np.isfinite(img).any() else 0.0
    return np.where(np.isfinite(img), img, fill)


def fit_poly(z, mask=None, order=1, max_pts=250_000):
    """Least-squares polynomial background. For large images the coefficients are
    fitted on a regular sub-grid (exact for 200/400 px frames), then evaluated on
    the full grid without building a full-size design matrix."""
    h, w = z.shape
    step = max(1, int(np.ceil(np.sqrt(h * w / max_pts))))
    xs = np.arange(0, w, step) / max(w - 1, 1) - 0.5
    ys = np.arange(0, h, step) / max(h - 1, 1) - 0.5
    zs = np.asarray(z[::step, ::step], dtype=np.float64)
    m = np.isfinite(zs)
    if mask is not None:
        mm = m & np.asarray(mask[::step, ::step], bool)
        n_terms = {0: 1, 1: 3, 2: 6}[order]
        if mm.sum() >= n_terms * 10:
            m = mm
    XX, YY = np.meshgrid(xs, ys)
    A = np.stack([t[m] for t in _poly_terms(XX, YY, order)], axis=1)
    coef, *_ = np.linalg.lstsq(A, zs[m], rcond=None)
    xf = (np.arange(w) / max(w - 1, 1) - 0.5)[None, :]
    yf = (np.arange(h) / max(h - 1, 1) - 0.5)[:, None]
    surf = np.zeros((h, w))
    for c, t in zip(coef, _poly_terms(xf, yf, order)):
        surf += c * t
    return surf, coef


def align_rows_median_diff(z):
    """Gwyddion-style 'median of differences' row alignment (robust to features)."""
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.nanmedian(np.diff(z, axis=0), axis=1)
    d = np.where(np.isfinite(d), d, 0.0)
    offs = np.concatenate([[0.0], np.cumsum(d)])
    return z - offs[:, None]


def background_mask(z, thr_nm):
    low = np.nanpercentile(z, 5)
    return z < (low + thr_nm)


def flatten(z, thr_nm=1.0):
    """Fixed flattening: plane -> row median-of-differences -> 2x (masked
    2nd-order background + masked row offsets) -> background median = 0.
    Returns flattened image (nm) and background mask."""
    z = z.astype(np.float64)
    if not np.isfinite(z).any():
        return z.astype(np.float32), np.zeros(z.shape, bool)
    plane, _ = fit_poly(z, None, 1)
    z = z - plane
    z = align_rows_median_diff(z)
    bg = background_mask(z, thr_nm)
    for _ in range(2):
        surf, _ = fit_poly(z, bg, 2)
        z = z - surf
        bg = background_mask(z, thr_nm)
        zb = np.where(bg, z, np.nan)
        cnt = bg.sum(axis=1)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row_off = np.nanmedian(zb, axis=1)
        row_off = np.where((cnt >= max(10, z.shape[1] // 20)) & np.isfinite(row_off),
                           row_off, 0.0)
        z = z - row_off[:, None]
        bg = background_mask(z, thr_nm)
    z = z - (np.nanmedian(z[bg]) if bg.any() else np.nanmedian(z))
    bg = background_mask(z, thr_nm)
    return z.astype(np.float32), bg


# ----------------------------------------------------------------------------
# Per-frame QC metrics
# ----------------------------------------------------------------------------
def dhash64(img):
    """64-bit difference hash of an image (as 16-char hex)."""
    from skimage.transform import resize
    small = resize(_filled(img), (8, 9), anti_aliasing=True, preserve_range=True)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return f"{v:016x}"


def thumb(img, size=32):
    from skimage.transform import resize
    t = resize(_filled(img), (size, size), anti_aliasing=True, preserve_range=True).astype(np.float32)
    return t


def _nanmed(a, axis=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def row_jump_count(z_raw, floor_nm=0.5, k=6.0):
    """Scan-line jumps: adjacent-row offsets (median of row differences after plane
    removal) that are both > k robust SDs and > floor_nm. Measured BEFORE row
    alignment, because flattening removes them."""
    if np.isfinite(z_raw).sum() < 100:
        return 0
    plane, _ = fit_poly(z_raw, None, 1)
    d = _nanmed(np.diff(z_raw - plane, axis=0), axis=1)
    d = d[np.isfinite(d)]
    if d.size < 5:
        return 0
    med = np.median(d)
    mad = np.median(np.abs(d - med)) * 1.4826
    return int(((np.abs(d - med) > k * mad) & (np.abs(d - med) > floor_nm)).sum())


def streak_row_count(z_flat, bg, k=6.0):
    """Rows whose background noise is abnormally high (horizontal streaks)."""
    zb = np.where(bg & np.isfinite(z_flat), z_flat, np.nan)
    cnt = np.isfinite(zb).sum(axis=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(zb, axis=1)
    sd = sd[cnt >= 10]
    if sd.size < 5:
        return 0
    med = np.median(sd)
    mad = np.median(np.abs(sd - med)) * 1.4826
    return int(((sd > med + k * max(mad, 1e-6)) & (sd > 2 * med)).sum())


def frame_metrics(z_raw, z_flat, bg, z_flat_retrace, sat, dead_frac, thr_nm):
    finite = np.isfinite(z_raw)
    m = {}
    m["unscanned_row_frac"] = float(dead_frac)
    # non-finite values other than the unscanned rows (true corruption)
    m["n_nonfinite"] = int((~finite).sum() - round(dead_frac * z_raw.shape[0]) * z_raw.shape[1])
    m["saturation_frac"] = float(sat)
    m["raw_mean_nm"] = float(np.nanmean(z_raw))
    m["raw_range_nm"] = float(np.nanmax(z_raw) - np.nanmin(z_raw))
    plane, coef = fit_poly(z_raw, None, 1)
    m["raw_tilt_x_nm"] = float(coef[1])
    m["raw_tilt_y_nm"] = float(coef[2])
    zb = z_flat[bg & np.isfinite(z_flat)]
    m["bg_frac"] = float(bg.mean())
    m["bg_rms_nm"] = float(np.sqrt(np.mean(zb ** 2))) if zb.size else float("nan")
    m["p99_nm"] = float(np.nanpercentile(z_flat, 99))
    m["coverage_proxy"] = float((z_flat[np.isfinite(z_flat)] > thr_nm).mean())  # QC proxy only, not a target
    m["n_line_jumps"] = row_jump_count(z_raw)
    m["n_streak_rows"] = streak_row_count(z_flat, bg)
    if z_flat_retrace is not None and z_flat_retrace.shape == z_flat.shape:
        m["trace_retrace_mad_nm"] = float(np.nanmedian(np.abs(z_flat - z_flat_retrace)))
    else:
        m["trace_retrace_mad_nm"] = float("nan")
    m["dhash"] = dhash64(z_flat)
    return m


# ----------------------------------------------------------------------------
# Worker: read + flatten + metrics for one frame
# ----------------------------------------------------------------------------
def _process_one(args):
    (row_id, kind, path, member, thr_nm) = args
    try:
        buf = read_bytes(kind, path, member)
        z, sat, dead = jpk_height(buf, retrace=False)
        z_level = float(np.nanmean(z))           # absolute z-piezo level (drift)
        z_raw = z - z_level
        z_flat, bg = flatten(z, thr_nm)
        try:
            zr, _, _ = jpk_height(buf, retrace=True)
            zr_flat, _ = flatten(zr, thr_nm)
        except Exception:
            zr_flat = None
        met = frame_metrics(z_raw, z_flat, bg, zr_flat, sat, dead, thr_nm)
        met["raw_mean_nm"] = z_level
        return row_id, z_raw.astype(np.float32), z_flat, thumb(z_flat), met, None
    except Exception as e:  # keep going; error is recorded in QC
        return row_id, None, None, None, None, f"{type(e).__name__}: {e}"


def _header_one(args):
    (row_id, kind, path, member) = args
    try:
        buf = read_bytes(kind, path, member)
        hdr = jpk_header(buf)
        return row_id, hdr, len(buf), None
    except Exception as e:
        return row_id, None, None, f"{type(e).__name__}: {e}"


def _pool(cfg):
    return ProcessPoolExecutor(max_workers=max(1, cfg.workers))


# ----------------------------------------------------------------------------
# STEP: verify
# ----------------------------------------------------------------------------
def md5_file(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def step_verify(cfg: Config):
    rows = []
    for name, expected in ZENODO_MD5.items():
        p = cfg.data / name
        if not p.exists():
            rows.append({"archive": name, "present": False, "md5": None,
                         "expected": expected, "match": None})
            continue
        t0 = time.time()
        got = md5_file(p)
        rows.append({"archive": name, "present": True, "md5": got, "expected": expected,
                     "match": got == expected, "size_bytes": p.stat().st_size,
                     "seconds": round(time.time() - t0, 1)})
        log.info("md5 %-20s %s  (%s)", name, "OK" if got == expected else "MISMATCH", got)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out / "checksums.csv", index=False)
    bad = df[(df.present) & (df.match == False)]  # noqa: E712
    if len(bad):
        log.warning("Checksum mismatch for: %s - re-download these archives.",
                    ", ".join(bad.archive))
    missing = df[~df.present].archive.tolist()
    if missing:
        log.warning("Archives not found as zip (fine if you extracted them): %s",
                    ", ".join(missing))
    return df


# ----------------------------------------------------------------------------
# STEP: index
# ----------------------------------------------------------------------------
def assign_sessions(df):
    """Session = contiguous block of frames with the same resolution and no gap
    longer than 60 s, within one concentration and regime."""
    df = df.sort_values(["regime_group", "concentration_nM", "timestamp"]).copy()
    sess = []
    for (_, conc), g in df.groupby(["regime_group", "concentration_nM"], sort=False):
        k = 0
        prev_t, prev_res = None, None
        for t, res in zip(g.timestamp, g.nx):
            if prev_t is not None and (res != prev_res or (t - prev_t).total_seconds() > 60):
                k += 1
            sess.append(k)
            prev_t, prev_res = t, res
    df["session_idx"] = sess
    return df


def step_index(cfg: Config):
    items = discover_sources(cfg.data)
    if not items:
        raise SystemExit(f"No .jpk frames found under {cfg.data}. Put the Zenodo zips "
                         f"(or extracted folders named like 2nM, 4nM ...) there.")
    if cfg.skip_overview:
        items = [it for it in items if not it[3]]
    base = pd.DataFrame(items, columns=["src_kind", "src_path", "member", "is_overview",
                                        "concentration_nM"])
    base["name"] = [m if isinstance(m, str) and m else p
                    for p, m in zip(base.src_path, base.member)]
    base["timestamp"] = [parse_ts(n) for n in base.name]
    n_bad_ts = int(base.timestamp.isna().sum())
    if n_bad_ts:
        log.warning("%d files without a timestamp in the name; header time will be used.",
                    n_bad_ts)
    base = base.reset_index(drop=True)
    base["row_id"] = np.arange(len(base))

    log.info("Reading headers of %d frames ...", len(base))
    jobs = [(r.row_id, r.src_kind, r.src_path,
             r.member if isinstance(r.member, str) else None) for r in base.itertuples()]
    hdrs = {}
    with _pool(cfg) as ex:
        for i, (rid, hdr, nbytes, err) in enumerate(ex.map(_header_one, jobs, chunksize=8)):
            hdrs[rid] = (hdr, nbytes, err)
            if (i + 1) % 500 == 0:
                log.info("  headers %d / %d", i + 1, len(jobs))
    recs = []
    for r in base.itertuples():
        hdr, nbytes, err = hdrs[r.row_id]
        rec = {"row_id": r.row_id, "file_bytes": nbytes, "header_error": err}
        if hdr:
            rec.update({k: hdr[k] for k in ("x0_m", "y0_m", "scan_u_m", "scan_v_m",
                                            "angle", "nx", "ny", "start_str", "end_str")})
            names = [c[1] for c in hdr["channels"]]
            rec["channels"] = ";".join(f"{c[1]}{'R' if c[2] else 'T'}" for c in hdr["channels"])
            rec["has_height"] = "height" in [str(n).lower() for n in names]
        recs.append(rec)
    df = base.merge(pd.DataFrame(recs), on="row_id")

    # header timestamp fallback, and acquisition duration of each frame
    def _hdr_time(s):
        if not isinstance(s, str):
            return None
        try:
            return datetime.strptime(s[:23], "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            return None
    ht = df.start_str.map(_hdr_time)
    df["timestamp"] = df.timestamp.where(df.timestamp.notna(), ht)
    et = df.end_str.map(_hdr_time)
    df["frame_duration_s"] = [(e - s).total_seconds() if (e is not None and s is not None)
                              else np.nan for s, e in zip(ht, et)]
    df = df[df.timestamp.notna()].copy()
    df["timestamp"] = pd.to_datetime(df.timestamp)

    df["nx"] = pd.to_numeric(df.nx, errors="coerce")
    df["ny"] = pd.to_numeric(df.ny, errors="coerce")
    df["scan_size_um"] = pd.to_numeric(df.scan_u_m, errors="coerce") * 1e6
    df["pixel_size_nm"] = df.scan_size_um * 1000.0 / df.nx
    df["regime_group"] = np.where(df.is_overview, "overview", "hsafm")
    df["regime"] = np.where(df.is_overview, "overview",
                            "hs" + df.nx.fillna(0).astype(int).astype(str))
    df = assign_sessions(df)

    # times relative to the first HS-AFM frame of each concentration
    df["experiment_id"] = "conc" + df.concentration_nM.astype("Int64").astype(str) + "nM"
    t0 = df[~df.is_overview].groupby("concentration_nM").timestamp.min()
    df["time_s"] = [(t - t0.get(c)).total_seconds() if c in t0.index else np.nan
                    for t, c in zip(df.timestamp, df.concentration_nM)]
    df = df.sort_values(["regime_group", "concentration_nM", "timestamp"]).reset_index(drop=True)
    df["frame_index"] = df.groupby(["regime_group", "concentration_nM"]).cumcount()
    df["session_id"] = (df.experiment_id + "_" + df.regime + "_s" +
                        df.session_idx.astype(str))
    df["frame_id"] = df.experiment_id + "_" + df.regime_group + "_" + \
        df.frame_index.map(lambda i: f"{i:05d}")
    df["dt_prev_s"] = df.groupby("session_id").timestamp.diff().dt.total_seconds()
    df["preprocess_version"] = PIPELINE_VERSION

    if cfg.max_frames > 0:
        df = df.groupby("session_id", group_keys=False).head(cfg.max_frames).reset_index(drop=True)
        log.warning("--max-frames %d: QUICK TEST MODE, not the full dataset.", cfg.max_frames)

    cols = ["frame_id", "experiment_id", "concentration_nM", "regime_group", "regime",
            "session_id", "frame_index", "timestamp", "time_s", "dt_prev_s",
            "frame_duration_s", "nx", "ny", "scan_size_um", "pixel_size_nm", "x0_m", "y0_m",
            "angle", "channels", "has_height", "file_bytes", "header_error",
            "src_kind", "src_path", "member", "preprocess_version"]
    df = df[cols]
    df.to_csv(cfg.out / "metadata" / "frames_index.csv", index=False)
    summ = (df.groupby(["experiment_id", "regime"]).agg(
        n_frames=("frame_id", "size"),
        n_sessions=("session_id", "nunique"),
        t_start_s=("time_s", "min"), t_end_s=("time_s", "max"),
        median_dt_s=("dt_prev_s", "median"),
        pixel_size_nm=("pixel_size_nm", "median")).reset_index())
    summ.to_csv(cfg.out / "metadata" / "experiment_summary.csv", index=False)
    log.info("Indexed %d frames:\n%s", len(df), summ.to_string(index=False))
    return df


# ----------------------------------------------------------------------------
# STEP: preprocess
# ----------------------------------------------------------------------------
def step_preprocess(cfg: Config):
    idx = pd.read_csv(cfg.out / "metadata" / "frames_index.csv")
    idx = idx[idx.has_height.fillna(False).astype(bool)].copy()
    stack_dir = cfg.out / "stacks"
    stack_dir.mkdir(parents=True, exist_ok=True)
    thumbs_all, metrics, errors = {}, [], []

    for sid, g in idx.groupby("session_id", sort=True):
        g = g.sort_values("frame_index")
        is_ov = (g.regime_group.iloc[0] == "overview")
        n, h, w = len(g), int(g.ny.iloc[0]), int(g.nx.iloc[0])
        log.info("Session %-32s %5d frames  %dx%d", sid, n, h, w)
        jobs = [(r.frame_id, r.src_kind, r.src_path,
                 None if pd.isna(r.member) else r.member, cfg.feature_thr_nm)
                for r in g.itertuples()]
        order = {fid: i for i, fid in enumerate(g.frame_id)}
        if is_ov:
            # overview images are large and few: store one file per image
            ov_dir = stack_dir / "overview"
            ov_dir.mkdir(exist_ok=True)
            for job in jobs:
                fid, zr, zf, th, met, err = _process_one(job)
                if err:
                    errors.append({"frame_id": fid, "error": err})
                    continue
                np.save(ov_dir / f"{fid}_flat.npy", zf)
                if cfg.save_raw:
                    np.save(ov_dir / f"{fid}_raw.npy", zr)
                met["frame_id"] = fid
                met["stack_file"] = f"stacks/overview/{fid}_flat.npy"
                met["stack_pos"] = -1
                metrics.append(met)
                thumbs_all[fid] = th
            continue
        flat_path = stack_dir / f"{sid}_flat.npy"
        flat = np.lib.format.open_memmap(flat_path, mode="w+", dtype=np.float32,
                                         shape=(n, h, w))
        flat[:] = np.nan
        raw = None
        if cfg.save_raw:
            raw = np.lib.format.open_memmap(stack_dir / f"{sid}_raw.npy", mode="w+",
                                            dtype=np.float32, shape=(n, h, w))
            raw[:] = np.nan
        done = 0
        with _pool(cfg) as ex:
            for fid, zr, zf, th, met, err in ex.map(_process_one, jobs, chunksize=4):
                done += 1
                if err:
                    errors.append({"frame_id": fid, "error": err})
                    continue
                i = order[fid]
                if zf.shape != (h, w):
                    errors.append({"frame_id": fid,
                                   "error": f"shape {zf.shape} != session {(h, w)}"})
                    continue
                flat[i] = zf
                if raw is not None:
                    raw[i] = zr
                met["frame_id"] = fid
                met["stack_file"] = f"stacks/{sid}_flat.npy"
                met["stack_pos"] = i
                metrics.append(met)
                thumbs_all[fid] = th
                if done % 200 == 0:
                    log.info("   %d / %d", done, n)
        flat.flush()
        del flat
        if raw is not None:
            raw.flush()
            del raw

    met = pd.DataFrame(metrics)
    met.to_csv(cfg.out / "metadata" / "frame_metrics.csv", index=False)
    pd.DataFrame(errors, columns=["frame_id", "error"]).to_csv(
        cfg.out / "metadata" / "read_errors.csv", index=False)
    if thumbs_all:
        ids = sorted(thumbs_all)
        np.savez_compressed(cfg.out / "metadata" / "thumbs32.npz",
                            frame_id=np.array(ids),
                            thumbs=np.stack([thumbs_all[i] for i in ids]))
    # per-experiment normalisation statistics (saved, NOT applied: in later portions
    # compute them on training experiments only)
    norm = {}
    for sid, g in idx.groupby("session_id"):
        f = cfg.out / "stacks" / f"{sid}_flat.npy"
        if not f.exists():
            continue
        a = np.load(f, mmap_mode="r")
        sample = np.asarray(a[:: max(1, len(a) // 50)]).ravel()
        sample = sample[np.isfinite(sample)]
        if sample.size:
            norm[sid] = {"median_nm": float(np.median(sample)),
                         "p1_nm": float(np.percentile(sample, 1)),
                         "p99_nm": float(np.percentile(sample, 99)),
                         "iqr_nm": float(np.subtract(*np.percentile(sample, [75, 25])))}
    with open(cfg.out / "metadata" / "normalization_stats_per_session.json", "w") as fh:
        json.dump(norm, fh, indent=2)
    log.info("Preprocessed %d frames, %d read errors.", len(met), len(errors))
    return met


# ----------------------------------------------------------------------------
# STEP: qc
# ----------------------------------------------------------------------------
def _robust_hi(x, k=5.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.inf
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return med + k * max(mad, 1e-6)


def hamming_hex(a, b):
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except (TypeError, ValueError):
        return 64


def step_qc(cfg: Config):
    from skimage.metrics import structural_similarity as ssim
    idx = pd.read_csv(cfg.out / "metadata" / "frames_index.csv")
    met = pd.read_csv(cfg.out / "metadata" / "frame_metrics.csv")
    err = pd.read_csv(cfg.out / "metadata" / "read_errors.csv")
    df = idx.merge(met, on="frame_id", how="left")
    df["read_error"] = df.frame_id.isin(err.frame_id) | ~df.has_height.fillna(False).astype(bool)
    df = df.sort_values(["session_id", "frame_index"]).reset_index(drop=True)

    # consecutive-frame similarity (SSIM on stored flattened frames) + dHash distance
    df["ssim_prev"] = np.nan
    df["dhash_dist_prev"] = np.nan
    for sid, g in df[~df.read_error & (df.stack_pos >= 0)].groupby("session_id"):
        f = cfg.out / "stacks" / f"{sid}_flat.npy"
        if not f.exists():
            continue
        a = np.load(f, mmap_mode="r")
        g = g.sort_values("stack_pos")
        prev, prev_hash = None, None
        for r in g.itertuples():
            cur = np.asarray(a[int(r.stack_pos)], dtype=np.float32)
            if prev is not None and np.isfinite(cur).all() and np.isfinite(prev).all():
                rng = float(max(np.percentile(cur, 99.5) - np.percentile(cur, 0.5), 1e-3))
                df.loc[r.Index, "ssim_prev"] = ssim(prev, cur, data_range=rng)
                df.loc[r.Index, "dhash_dist_prev"] = hamming_hex(prev_hash, r.dhash)
            prev, prev_hash = cur, r.dhash

    # thresholds are robust statistics within each session
    flags = []
    for sid, g in df.groupby("session_id"):
        thr_noise = _robust_hi(g.bg_rms_nm)
        thr_tr = _robust_hi(g.trace_retrace_mad_nm)
        med_dt = np.nanmedian(g.dt_prev_s) if g.dt_prev_s.notna().any() else np.nan
        for r in g.itertuples():
            f = {"frame_id": r.frame_id}
            f["flag_read_error"] = bool(r.read_error)
            f["flag_nonfinite"] = bool((r.n_nonfinite or 0) > 0) if pd.notna(r.n_nonfinite) else False
            f["flag_incomplete_scan"] = bool(pd.notna(r.unscanned_row_frac) and
                                             r.unscanned_row_frac > 0)
            f["flag_saturated"] = bool(pd.notna(r.saturation_frac) and r.saturation_frac > 0.01)
            f["flag_line_artifacts"] = bool(
                (pd.notna(r.n_line_jumps) and r.n_line_jumps > 3) or
                (pd.notna(r.n_streak_rows) and r.n_streak_rows > 0.05 * (r.ny or 200)))
            f["flag_trace_retrace"] = bool(pd.notna(r.trace_retrace_mad_nm) and
                                           r.trace_retrace_mad_nm > thr_tr)
            f["flag_noisy"] = bool(pd.notna(r.bg_rms_nm) and r.bg_rms_nm > thr_noise)
            f["flag_duplicate"] = bool(pd.notna(r.ssim_prev) and r.ssim_prev > 0.995 and
                                       pd.notna(r.dhash_dist_prev) and r.dhash_dist_prev <= 2)
            f["flag_time_gap"] = bool(pd.notna(r.dt_prev_s) and np.isfinite(med_dt) and
                                      r.dt_prev_s > 3 * med_dt)
            flags.append(f)
    fl = pd.DataFrame(flags)
    df = df.merge(fl, on="frame_id", how="left")
    hard = ["flag_read_error", "flag_nonfinite", "flag_incomplete_scan", "flag_saturated",
            "flag_duplicate"]
    soft = ["flag_line_artifacts", "flag_trace_retrace", "flag_noisy", "flag_time_gap"]
    df[hard + soft] = df[hard + soft].fillna(False).astype(bool)
    df["exclude"] = df[hard].any(axis=1)
    df["n_soft_flags"] = df[soft].sum(axis=1)
    df["quality_flags"] = df[hard + soft].apply(
        lambda r: "|".join(c.replace("flag_", "") for c in hard + soft if r[c]), axis=1)
    df.to_csv(cfg.out / "metadata" / "frames_qc.csv", index=False)

    tab = df.groupby(["experiment_id", "regime"]).agg(
        n=("frame_id", "size"), excluded=("exclude", "sum"),
        **{c.replace("flag_", ""): (c, "sum") for c in hard + soft}).reset_index()
    tab.to_csv(cfg.out / "metadata" / "qc_summary.csv", index=False)
    log.info("QC summary:\n%s", tab.to_string(index=False))
    return df


# ----------------------------------------------------------------------------
# STEP: splits (experiment level) + leakage checks
# ----------------------------------------------------------------------------
def _split_record(name, desc, train, val, test, df):
    def ids(cs):
        return df[df.concentration_nM.isin(cs)].frame_id.tolist()
    return {"name": name, "description": desc, "unit": "experiment (one movie per concentration)",
            "train_conc_nM": train, "val_conc_nM": val, "test_conc_nM": test,
            "train": ids(train), "val": ids(val), "test": ids(test)}


def temporal_split(df, train_frac=0.6, val_frac=0.15, gap_frames=16):
    """Within-movie early/late split with gaps of `gap_frames` between blocks, so no
    clip of up to `gap_frames` frames crosses a boundary. Secondary analysis only:
    train/test frames come from the same movie."""
    tr, va, te = [], [], []
    for _, g in df.sort_values("time_s").groupby("experiment_id"):
        n = len(g)
        a = int(n * train_frac)
        b = int(n * (train_frac + val_frac))
        ids = g.frame_id.tolist()
        tr += ids[: max(0, a - gap_frames)]
        va += ids[a: max(a, b - gap_frames)]
        te += ids[b:]
    return {"name": "T1_temporal_early_late",
            "description": f"Per movie: first {int(train_frac*100)}% train, next "
                           f"{int(val_frac*100)}% val, rest test, {gap_frames}-frame gaps. "
                           f"Secondary (same movie in train and test).",
            "unit": "time block within movie", "train": tr, "val": va, "test": te}


def check_split(sp, df):
    s_tr, s_va, s_te = set(sp["train"]), set(sp["val"]), set(sp["test"])
    out = {"n_train": len(s_tr), "n_val": len(s_va), "n_test": len(s_te),
           "frame_overlap": len(s_tr & s_te) + len(s_tr & s_va) + len(s_va & s_te)}
    e = df.set_index("frame_id").experiment_id
    ex = {k: set(e.reindex(list(v)).dropna()) for k, v in
          (("train", s_tr), ("val", s_va), ("test", s_te))}
    out["experiment_overlap_train_test"] = sorted(ex["train"] & ex["test"])
    out["experiment_overlap_train_val"] = sorted(ex["train"] & ex["val"])
    return out


def step_splits(cfg: Config):
    df = pd.read_csv(cfg.out / "metadata" / "frames_qc.csv")
    hs = df[(df.regime_group == "hsafm") & (~df.exclude)].copy()
    present = sorted(int(c) for c in hs.concentration_nM.dropna().unique())
    log.info("Concentrations available for splits: %s", present)
    splits = []
    # LOCO: test = one concentration, val = neighbouring concentration, train = rest
    for c in present:
        others = [x for x in present if x != c]
        if not others:
            continue
        val = [min(others, key=lambda x: (abs(x - c), x))]
        train = [x for x in others if x not in val]
        if not train:
            continue
        splits.append(_split_record(f"LOCO_{c}nM", f"Leave-one-concentration-out, test {c} nM",
                                    train, val, [c], hs))
    def _maybe(name, desc, tr, va, te):
        if all(x in present for x in tr + va + te):
            splits.append(_split_record(name, desc, tr, va, te, hs))
        else:
            log.warning("Split %s skipped: needs %s nM", name, sorted(tr + va + te))
    _maybe("P1_high_OOD", "Train low, test high concentrations", [2, 4, 6], [8], [10, 12])
    _maybe("P2_low_OOD", "Train high, test low concentrations", [8, 10, 12], [6], [2, 4])
    splits.append(temporal_split(hs))

    sdir = cfg.out / "splits"
    sdir.mkdir(exist_ok=True)
    checks = {}
    for sp in splits:
        ck = check_split(sp, hs)
        checks[sp["name"]] = ck
        sp_out = dict(sp, checks=ck, pipeline_version=PIPELINE_VERSION,
                      excluded_frames_removed=True)
        with open(sdir / f"{sp['name']}.json", "w") as fh:
            json.dump(sp_out, fh)
        ok = ck["frame_overlap"] == 0 and (sp["name"].startswith("T1") or
                                           not ck["experiment_overlap_train_test"])
        log.info("Split %-24s train %5d val %5d test %5d  %s", sp["name"], ck["n_train"],
                 ck["n_val"], ck["n_test"], "PASS" if ok else "FAIL")
        if not ok:
            raise RuntimeError(f"Leakage check failed for {sp['name']}: {ck}")
    with open(sdir / "split_checks.json", "w") as fh:
        json.dump(checks, fh, indent=2)
    # overview images are kept out of all temporal splits
    df[df.regime_group == "overview"][["frame_id", "concentration_nM"]].to_csv(
        sdir / "overview_context_only.csv", index=False)
    return splits


# ----------------------------------------------------------------------------
# STEP: audit - nearest-neighbour similarity between train and test frames
# ----------------------------------------------------------------------------
def _load_thumbs(cfg):
    z = np.load(cfg.out / "metadata" / "thumbs32.npz", allow_pickle=False)
    ids = z["frame_id"].astype(str)
    T = z["thumbs"].reshape(len(ids), -1).astype(np.float64)
    T = T - T.mean(1, keepdims=True)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    return {k: i for i, k in enumerate(ids)}, T


def _max_sim(A, B, chunk=2048):
    out = np.full(len(A), -1.0)
    for i in range(0, len(A), chunk):
        out[i:i + chunk] = (A[i:i + chunk] @ B.T).max(axis=1)
    return out


def step_audit(cfg: Config):
    pos, T = _load_thumbs(cfg)
    qc = pd.read_csv(cfg.out / "metadata" / "frames_qc.csv").set_index("frame_id")
    rows, per_frame = [], {}
    for f in sorted((cfg.out / "splits").glob("*.json")):
        if f.name == "split_checks.json":
            continue
        sp = json.load(open(f))
        tr = [pos[i] for i in sp["train"] if i in pos]
        te_ids = [i for i in sp["test"] if i in pos]
        te = [pos[i] for i in te_ids]
        if not tr or not te:
            continue
        s_tt = _max_sim(T[te], T[tr])
        # reference: similarity of each test frame to the nearest test frame that is
        # at least 16 frames away in the same movie (what "independent" looks like)
        ref = np.full(len(te), np.nan)
        exp = qc.experiment_id.reindex(te_ids).values
        fi = qc.frame_index.reindex(te_ids).values
        Tte = T[te]
        for e in np.unique(exp):
            m = np.where(exp == e)[0]
            S = Tte[m] @ Tte[m].T
            far = np.abs(fi[m][:, None] - fi[m][None, :]) >= 16
            S = np.where(far, S, -np.inf)
            mx = S.max(axis=1)
            ref[m] = np.where(np.isfinite(mx), mx, np.nan)
        rows.append({"split": sp["name"], "n_test": len(te),
                     "test_to_train_max_sim_median": float(np.median(s_tt)),
                     "test_to_train_max_sim_p95": float(np.percentile(s_tt, 95)),
                     "frac_test_sim_gt_0.98": float((s_tt > 0.98).mean()),
                     "within_test_far_sim_median": float(np.nanmedian(ref))})
        per_frame[sp["name"]] = {"test_to_train": s_tt.tolist(),
                                 "within_test_far": ref.tolist()}
    res = pd.DataFrame(rows, columns=["split", "n_test", "test_to_train_max_sim_median",
                                      "test_to_train_max_sim_p95", "frac_test_sim_gt_0.98",
                                      "within_test_far_sim_median"])
    res.to_csv(cfg.out / "metadata" / "leakage_audit.csv", index=False)
    with open(cfg.out / "metadata" / "leakage_audit_values.json", "w") as fh:
        json.dump(per_frame, fh)
    if len(res):
        log.info("Leakage audit (cosine similarity of 32x32 thumbnails):\n%s",
                 res.to_string(index=False))
    return res


# ----------------------------------------------------------------------------
# STEP: figures
# ----------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
        "grid.linewidth": 0.5, "axes.edgecolor": "#808080", "lines.linewidth": 1.2,
        "savefig.dpi": 300, "figure.dpi": 100, "pdf.fonttype": 42, "axes.axisbelow": True})
    return plt


def conc_colors(concs):
    """Concentration is ordered -> one sequential, colour-blind-safe map (cividis),
    avoiding its lightest end so every step stays visible on white."""
    import matplotlib
    cmap = matplotlib.colormaps["cividis"]
    concs = sorted(concs)
    if len(concs) == 1:
        return {concs[0]: cmap(0.4)}
    return {c: cmap(0.0 + 0.62 * i / (len(concs) - 1)) for i, c in enumerate(concs)}


def _plot_by_session(ax, g, ycol, color, label=None, lw=0.8):
    """Plot y vs time per session so lines never bridge acquisition gaps."""
    first = True
    for _, gs in g.sort_values("time_s").groupby("session_id"):
        if gs[ycol].notna().any():
            ax.plot(gs.time_s, gs[ycol], color=color, lw=lw,
                    label=label if first else None)
            first = False


def _save(fig, out, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def step_figures(cfg: Config):
    plt = _mpl()
    fdir = cfg.out / "figures"
    fdir.mkdir(exist_ok=True)
    df = pd.read_csv(cfg.out / "metadata" / "frames_qc.csv")
    hs = df[df.regime_group == "hsafm"].copy()
    concs = sorted(int(c) for c in hs.concentration_nM.dropna().unique())
    col = conc_colors(concs)
    made = []

    # Fig A - acquisition timeline
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    for i, c in enumerate(concs):
        g = hs[hs.concentration_nM == c]
        for reg, mk in (("hs200", "|"), ("hs400", "|")):
            gg = g[g.regime == reg]
            if len(gg):
                ax.scatter(gg.time_s, np.full(len(gg), i) + (0.15 if reg == "hs400" else -0.15),
                           marker=mk, s=12, color=col[c], linewidths=0.6)
        ex = g[g.exclude]
        if len(ex):
            ax.scatter(ex.time_s, np.full(len(ex), i) + 0.38, marker="v", s=8,
                       color="#c0392b", label="excluded" if i == 0 else None)
    ax.set_yticks(range(len(concs)))
    ax.set_yticklabels([f"{c} nM" for c in concs])
    ax.set_xlabel("Time since first frame of movie (s)")
    ax.set_title("Acquisition timeline (lower ticks: 200×200 px, 1 fps; upper: 400×400 px, 0.5 fps)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(concs) - 0.4)
    if hs.exclude.any():
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    _save(fig, fdir, "figA_acquisition_timeline"); made.append("figA_acquisition_timeline")

    # Fig B - frames per experiment and regime (+ excluded)
    tab = hs.groupby(["concentration_nM", "regime"]).agg(n=("frame_id", "size"),
                                                         ex=("exclude", "sum")).reset_index()
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    x = np.arange(len(concs))
    bottom = np.zeros(len(concs))
    shades = {"hs200": "#4a5a7a", "hs400": "#a3acc0"}
    for reg in ("hs200", "hs400"):
        v = np.array([tab[(tab.concentration_nM == c) & (tab.regime == reg)].n.sum() for c in concs])
        ax.bar(x, v, bottom=bottom, width=0.6, color=shades[reg], edgecolor="white",
               linewidth=1, label={"hs200": "200×200 px", "hs400": "400×400 px"}[reg])
        bottom += v
    for xi, tot, c in zip(x, bottom, concs):
        nex = int(hs[(hs.concentration_nM == c)].exclude.sum())
        ax.text(xi, tot + bottom.max() * 0.02, f"{int(tot)}" + (f"\n({nex} excl.)" if nex else ""),
                ha="center", va="bottom", fontsize=6.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} nM" for c in concs])
    ax.set_ylabel("HS-AFM frames")
    ax.set_ylim(0, bottom.max() * 1.25 + 1)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Frames per experiment (one movie per concentration)")
    ax.grid(axis="x", visible=False)
    _save(fig, fdir, "figB_frames_per_experiment"); made.append("figB_frames_per_experiment")

    # Fig C - raw -> flattened -> robust-normalised example per concentration
    def _get(row, kind):
        if row.stack_pos < 0:
            return np.load(cfg.out / str(row.stack_file).replace("_flat", f"_{kind}"))
        f = cfg.out / "stacks" / f"{row.session_id}_{kind}.npy"
        if not f.exists():
            return None
        return np.asarray(np.load(f, mmap_mode="r")[int(row.stack_pos)])
    ok = hs[~hs.exclude & hs.stack_pos.notna()]
    if len(ok):
        fig, axes = plt.subplots(3, len(concs), figsize=(1.35 * len(concs) + 0.6, 4.4),
                                 squeeze=False)
        for j, c in enumerate(concs):
            g = ok[ok.concentration_nM == c].sort_values("time_s")
            if not len(g):
                continue
            r = g.iloc[len(g) // 2]
            imgs = [_get(r, "raw"), _get(r, "flat")]
            fl = imgs[1]
            if fl is not None:
                med = np.nanmedian(fl)
                iqr = np.subtract(*np.nanpercentile(fl, [75, 25])) + 1e-6
                imgs.append((fl - med) / iqr)
            else:
                imgs.append(None)
            for i, (im, lab) in enumerate(zip(imgs, ["plane-levelled", "flattened (nm)",
                                                       "robust-normalised"])):
                a = axes[i, j]
                a.axis("off")
                if im is None:
                    continue
                lo, hi = np.nanpercentile(im, [1, 99.5])
                h_ = a.imshow(im, cmap="afmhot", vmin=lo, vmax=hi, interpolation="nearest")
                if i == 0:
                    a.set_title(f"{c} nM\nt = {r.time_s:.0f} s", fontsize=7)
                if j == 0:
                    a.text(-0.08, 0.5, lab, transform=a.transAxes, rotation=90,
                           ha="right", va="center", fontsize=7)
                if i == 1:
                    cb = fig.colorbar(h_, ax=a, fraction=0.046, pad=0.02)
                    cb.ax.tick_params(labelsize=5)
        fig.suptitle("Preprocessing: same frame at each stage (scan 1×1 µm²)", fontsize=8)
        _save(fig, fdir, "figC_preprocessing_examples"); made.append("figC_preprocessing_examples")

    # Fig D - drift / QC metrics vs time
    met = [("raw_mean_nm", "Mean z level (nm)"), ("bg_rms_nm", "Background RMS (nm)"),
           ("trace_retrace_mad_nm", "Trace–retrace MAD (nm)"),
           ("coverage_proxy", "Feature-pixel fraction (QC proxy)")]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2), sharex=True)
    for a, (m, lab) in zip(axes.ravel(), met):
        for c in concs:
            _plot_by_session(a, hs[hs.concentration_nM == c], m, col[c], f"{c} nM")
        a.set_ylabel(lab)
    for a in axes[1]:
        a.set_xlabel("Time (s)")
    fig.suptitle("Drift and quality over time (lines break at acquisition gaps)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(concs),
               frameon=False, fontsize=7)
    _save(fig, fdir, "figD_drift_quality_vs_time"); made.append("figD_drift_quality_vs_time")

    # Fig E - QC flag counts
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    cnt = hs.groupby("concentration_nM")[flag_cols].sum().reindex(concs)
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    im = ax.imshow(cnt.T.values, aspect="auto", cmap="Greys", vmin=0,
                   vmax=max(1, cnt.values.max()))
    ax.set_xticks(range(len(concs)))
    ax.set_xticklabels([f"{c} nM" for c in concs])
    ax.set_yticks(range(len(flag_cols)))
    ax.set_yticklabels([c.replace("flag_", "").replace("_", " ") for c in flag_cols])
    for i in range(len(flag_cols)):
        for j in range(len(concs)):
            v = int(cnt.T.values[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5,
                    color="white" if v > cnt.values.max() * 0.6 else "#222222")
    ax.grid(False)
    ax.set_title("Quality flags per experiment (frames)")
    _save(fig, fdir, "figE_qc_flags"); made.append("figE_qc_flags")

    # Fig F - artifact gallery: worst frame for each soft flag
    gal, used = [], set()
    for fc, metric in (("flag_line_artifacts", "n_line_jumps"),
                       ("flag_trace_retrace", "trace_retrace_mad_nm"),
                       ("flag_noisy", "bg_rms_nm")):
        g = hs[hs[fc] & hs.stack_pos.notna() & (hs.stack_pos >= 0)].sort_values(metric, ascending=False)
        g = g[~g.frame_id.isin(used)]
        for r in g.head(2).itertuples():
            gal.append((fc.replace("flag_", "").replace("_", " "), r))
            used.add(r.frame_id)
    if gal:
        fig, axes = plt.subplots(1, len(gal), figsize=(1.5 * len(gal), 1.9), squeeze=False)
        for a, (lab, r) in zip(axes[0], gal):
            im = _get(r, "flat")
            a.axis("off")
            if im is None:
                continue
            lo, hi = np.nanpercentile(im, [1, 99.5])
            a.imshow(im, cmap="afmhot", vmin=lo, vmax=hi, interpolation="nearest")
            a.set_title(f"{lab}\n{int(r.concentration_nM)} nM, t={r.time_s:.0f}s", fontsize=6)
        fig.suptitle("Examples of flagged frames (kept, soft flags)", fontsize=8)
        _save(fig, fdir, "figF_artifact_gallery"); made.append("figF_artifact_gallery")

    # Fig G - consecutive-frame SSIM and frame-interval distribution
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
    for c in concs:
        _plot_by_session(axes[0], hs[hs.concentration_nM == c], "ssim_prev", col[c],
                         f"{c} nM", lw=0.7)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("SSIM to previous frame")
    dts = hs.dt_prev_s.dropna()
    dts = dts[dts < 10]
    axes[1].hist(dts, bins=60, color="#4a5a7a")
    axes[1].set_xlabel("Interval to previous frame (s)")
    axes[1].set_ylabel("Frames")
    axes[1].set_title("Frame intervals (gaps > 10 s not shown)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=len(concs), frameon=False, fontsize=7)
    _save(fig, fdir, "figG_temporal_similarity_intervals"); made.append("figG_temporal_similarity_intervals")

    # Fig H - leakage audit
    lv = cfg.out / "metadata" / "leakage_audit_values.json"
    if lv.exists():
        vals = json.load(open(lv))
        names = sorted([n for n in vals if n.startswith("LOCO")],
                       key=lambda n: int(re.search(r"(\d+)nM", n).group(1))) + \
            sorted(n for n in vals if not n.startswith("LOCO"))
        if names:
            fig, ax = plt.subplots(figsize=(6.5, 2.6))
            data_tt = [np.array(vals[n]["test_to_train"]) for n in names]
            data_ref = [np.array([v for v in vals[n]["within_test_far"] if v is not None and np.isfinite(v)]) for n in names]
            pos_ = np.arange(len(names))
            b1 = ax.boxplot(data_tt, positions=pos_ - 0.18, widths=0.3, showfliers=False,
                            patch_artist=True)
            b2 = ax.boxplot([d if len(d) else np.array([np.nan]) for d in data_ref],
                            positions=pos_ + 0.18, widths=0.3, showfliers=False,
                            patch_artist=True)
            for b, cc in ((b1, "#4a5a7a"), (b2, "#c9cdd6")):
                for p in b["boxes"]:
                    p.set_facecolor(cc); p.set_edgecolor("#333333"); p.set_linewidth(0.6)
                for k in ("whiskers", "caps", "medians"):
                    for l in b[k]:
                        l.set_color("#333333"); l.set_linewidth(0.6)
            ax.set_xticks(pos_)
            short = {"P1_high_OOD": "P1\nhigh OOD", "P2_low_OOD": "P2\nlow OOD",
                     "T1_temporal_early_late": "T1\nearly→late"}
            ax.set_xticklabels([short.get(n, n.replace("LOCO_", "LOCO\n")) for n in names],
                               fontsize=6.5)
            ax.set_ylabel("Max cosine similarity")
            ax.legend([b1["boxes"][0], b2["boxes"][0]],
                      ["test → nearest train frame", "test → nearest test frame ≥16 frames away"],
                      frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
            ax.set_title("Leakage audit: train–test similarity (32×32 thumbnails)")
            ax.grid(axis="x", visible=False)
            _save(fig, fdir, "figH_leakage_audit"); made.append("figH_leakage_audit")

    # Fig I - pixel-size / resolution table as a figure-ready CSV
    log.info("Figures written: %s", ", ".join(made))
    return made


# ----------------------------------------------------------------------------
# STEP: report
# ----------------------------------------------------------------------------
def step_report(cfg: Config):
    md = cfg.out / "metadata"
    df = pd.read_csv(md / "frames_qc.csv")
    hs = df[df.regime_group == "hsafm"]
    ov = df[df.regime_group == "overview"]
    per = hs.groupby("experiment_id").agg(
        concentration_nM=("concentration_nM", "first"), frames=("frame_id", "size"),
        excluded=("exclude", "sum"), sessions=("session_id", "nunique"),
        duration_s=("time_s", "max"),
        frames_200px=("regime", lambda s: int((s == "hs200").sum())),
        frames_400px=("regime", lambda s: int((s == "hs400").sum()))).reset_index()
    per = per.sort_values("concentration_nM")
    per.to_csv(md / "table1_dataset_composition.csv", index=False)
    checks = json.load(open(cfg.out / "splits" / "split_checks.json")) \
        if (cfg.out / "splits" / "split_checks.json").exists() else {}
    cs = pd.read_csv(cfg.out / "checksums.csv") if (cfg.out / "checksums.csv").exists() else None
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "quick_test_mode": cfg.max_frames > 0,
        "n_experiments": int(hs.experiment_id.nunique()),
        "n_hsafm_frames": int(len(hs)), "n_hsafm_excluded": int(hs.exclude.sum()),
        "n_overview_images": int(len(ov)),
        "published_analyzed_frames_reference": PUBLISHED_TOTAL_ANALYZED,
        "frames_per_experiment": per.set_index("experiment_id").frames.to_dict(),
        "checksums_ok": None if cs is None else bool(cs[cs.present].match.all()),
        "split_checks": checks,
        "flattening": {"protocol": "plane(1) -> row median-of-differences -> 2x[masked "
                                   "poly(2) + masked row medians] -> background median=0",
                       "feature_threshold_nm": cfg.feature_thr_nm},
        "note_time_origin": "time_s = seconds since the first recorded frame of each "
                            "movie, NOT since injection (injection time is not in the data).",
    }
    with open(cfg.out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    lines = [f"# Portion 1 report ({PIPELINE_VERSION})", "",
             f"- Experiments (movies): **{summary['n_experiments']}** - one per concentration.",
             f"- HS-AFM frames: **{summary['n_hsafm_frames']}** "
             f"({summary['n_hsafm_excluded']} excluded by hard QC flags).",
             f"- Overview images: {summary['n_overview_images']} (context only, not in splits).",
             f"- Published analysed total for reference: {PUBLISHED_TOTAL_ANALYZED}.",
             f"- Checksums OK: {summary['checksums_ok']}", ""]
    if summary["quick_test_mode"]:
        lines.insert(2, "> **QUICK TEST MODE** (--max-frames): numbers are not the full dataset.\n")
    lines += ["## Table 1 - dataset composition", "", per.to_markdown(index=False)
              if _has_tabulate() else per.to_string(index=False), ""]
    lines += ["## Split checks", ""]
    for k, v in checks.items():
        lines.append(f"- {k}: train {v['n_train']}, val {v['n_val']}, test {v['n_test']}, "
                     f"frame overlap {v['frame_overlap']}, "
                     f"experiment overlap train/test {v['experiment_overlap_train_test']}")
    la = md / "leakage_audit.csv"
    if la.exists() and la.stat().st_size > 0:
        lad = pd.read_csv(la)
        if len(lad):
            lines += ["", "## Leakage audit (cosine similarity, 32x32 thumbnails)", "",
                      lad.to_string(index=False)]
    (cfg.out / "summary.md").write_text("\n".join(lines))
    log.info("Report written: %s", cfg.out / "summary.md")
    return summary


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
STEPS = ["verify", "index", "preprocess", "qc", "splits", "audit", "figures", "report"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="folder with the Zenodo zips and/or extracted folders")
    ap.add_argument("--out", required=True, type=Path, help="output folder")
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--max-frames", type=int, default=0,
                    help="quick test: only the first N frames of each session")
    ap.add_argument("--feature-thr-nm", type=float, default=1.0)
    ap.add_argument("--no-raw", action="store_true", help="do not save unflattened stacks")
    ap.add_argument("--skip-overview", action="store_true",
                    help="ignore the overview archive (large files, not used in splits)")
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "metadata").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.WARNING)   # third-party libraries: warnings only
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.handlers.clear()
    for hnd in (logging.StreamHandler(sys.stdout), logging.FileHandler(a.out / "pipeline.log")):
        hnd.setFormatter(fmt)
        log.addHandler(hnd)
    log.setLevel(logging.INFO)
    log.propagate = False
    if not a.data.exists():
        raise SystemExit(f"--data folder not found: {a.data}")
    cfg = Config(data=a.data, out=a.out, workers=a.workers, max_frames=a.max_frames,
                 feature_thr_nm=a.feature_thr_nm, save_raw=not a.no_raw,
                 skip_overview=a.skip_overview)
    with open(a.out / "config.json", "w") as fh:
        json.dump({**{k: str(v) for k, v in vars(cfg).items()},
                   "pipeline_version": PIPELINE_VERSION, "steps": a.steps}, fh, indent=2)
    funcs = {"verify": step_verify, "index": step_index, "preprocess": step_preprocess,
             "qc": step_qc, "splits": step_splits, "audit": step_audit,
             "figures": step_figures, "report": step_report}
    for s in STEPS:
        if s in a.steps:
            t0 = time.time()
            log.info("=== %s ===", s)
            funcs[s](cfg)
            log.info("=== %s done in %.1f s ===", s, time.time() - t0)


if __name__ == "__main__":
    main()
