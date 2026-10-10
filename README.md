# Portion 1 — Data audit, preprocessing & leakage control (CPU)

Dataset: Zenodo 10.5281/zenodo.14275812 (Pothineni, Barner, Keller), CC-BY 4.0.

## What the data really looks like (checked on the real archives)

| Item | Finding |
|---|---|
| File format | JPK `.jpk` (TIFF container), **one frame per file**, timestamp in the file name |
| Channels per frame | amplitude, phase, **height** — each as trace and retrace. The pipeline uses **height, trace**, converted to nm with the calibration stored in the file |
| Experiments | **One movie per concentration.** 6 nM, for example, has one folder for the 200 px session and one for the 400 px session of the *same* experiment |
| Scan size | 1×1 µm² (HS-AFM frames) → 5 nm/px at 200 px, 2.5 nm/px at 400 px |
| Incomplete frames | Some frames stop early: the unscanned rows are constant. They are detected, set to NaN and the frame is excluded |
| Overview archive | Folders 4, 6, 8, 10, 12 nM (no 2 nM), 2048 px and larger; used as context only |
| Frames per archive | 2 nM 598 (200 px only, no 400 px session) · 4 nM 888 · 6 nM 799 · 8 nM 863 · 10 nM 903 · 12 nM 884 → **4,935** HS-AFM frames (the paper analysed 4,267) |
| z drift | The 2 nM movie drifts ~500 nm in z over 10 min; flattening removes it, and Fig D documents it |

Expected run time on 2 CPUs: about 10–15 min for the HS-AFM frames plus ~30 s per overview image. Stacks need ~4–5 GB of disk with raw stacks, about half that with `--no-raw`.

## Files

| File | Purpose |
|---|---|
| `portion1_pipeline.py` | The whole pipeline (one file, CPU only) |
| `Portion1_Colab.ipynb` | Colab notebook: mount Drive → quick test → full run → copy results to Drive |
| `requirements.txt` | Python packages |

## Run it

**Colab (recommended):** upload `portion1_pipeline.py` to Drive, open the notebook, edit the paths in cell 2, run all cells. Choose a CPU runtime.

**Any machine:**
```bash
pip install -r requirements.txt
# 2-minute smoke test first
python portion1_pipeline.py --data /path/to/zips --out p1_test --max-frames 15 --skip-overview
# full run
python portion1_pipeline.py --data /path/to/zips --out p1_output
# re-run selected steps only
python portion1_pipeline.py --data /path/to/zips --out p1_output --steps qc splits audit figures report
```
`--data` may hold the original zips, folders extracted from them (named `2nM`, `4nM`, …, `overview4x4um`), or both. The input is never modified.

Useful options: `--workers N` (default: all CPUs), `--no-raw` (skip unflattened stacks, halves disk use), `--skip-overview`, `--feature-thr-nm 1.0`.

## Steps and outputs

| Step | Output |
|---|---|
| `verify` | `checksums.csv` — md5 of each zip vs the Zenodo record |
| `index` | `metadata/frames_index.csv` — one row per frame: experiment_id, concentration_nM, regime (hs200 / hs400 / overview), session_id, frame_index, timestamp, **time_s**, dt_prev_s, nx, ny, scan_size_um, **pixel_size_nm**, preprocess_version; `metadata/experiment_summary.csv` |
| `preprocess` | `stacks/<session>_flat.npy` (N×H×W float32, nm, flattened) and `_raw.npy` (plane-unlevelled, mean-removed); `stacks/overview/*.npy`; `metadata/frame_metrics.csv`; `metadata/thumbs32.npz`; `metadata/normalization_stats_per_session.json` |
| `qc` | `metadata/frames_qc.csv` (all metadata + QC metrics + flags + `exclude`); `metadata/qc_summary.csv` |
| `splits` | `splits/LOCO_<c>nM.json` ×6, `P1_high_OOD.json`, `P2_low_OOD.json`, `T1_temporal_early_late.json`, `split_checks.json`, `overview_context_only.csv` |
| `audit` | `metadata/leakage_audit.csv` — nearest-neighbour similarity, test→train vs within-test |
| `figures` | `figures/figA…figH` as PNG (300 dpi) and PDF |
| `report` | `summary.md`, `summary.json`, `metadata/table1_dataset_composition.csv` |

### Load a session in later portions
```python
import numpy as np, pandas as pd, json
qc = pd.read_csv("p1_output/metadata/frames_qc.csv")
split = json.load(open("p1_output/splits/LOCO_6nM.json"))
stack = np.load("p1_output/stacks/conc6nM_hs200_s0_flat.npy", mmap_mode="r")
row = qc[qc.frame_id == split["test"][0]].iloc[0]
frame = stack[int(row.stack_pos)]          # nm, flattened
```

## Fixed preprocessing protocol (version `p1-1.0.0`)
1. Height trace → nm using the file's calibration slot (`calibrated`).
2. Unscanned rows → NaN.
3. First-order plane subtraction.
4. Row alignment by median of differences between adjacent rows.
5. Twice: second-order background fit on background pixels only, then per-row offset from background pixels. Background = pixels below (5th percentile + 1 nm).
6. Background median set to 0 nm.

Normalisation statistics are **saved but not applied**. In later portions, compute them on training experiments only.

## QC flags
**Hard (frame excluded):** read error, non-finite values, incomplete scan, saturation > 1 %, duplicate of previous frame (SSIM > 0.995 and dHash distance ≤ 2).
**Soft (frame kept, flagged):** line artifacts (> 3 row jumps), trace/retrace mismatch, high background noise (both robust: > median + 5·MAD within the session), time gap (> 3× the session's median interval).

Excluded frames are listed, never silently deleted.

## Notes for the paper
- `time_s` counts from the first recorded frame of each movie, **not** from injection (the injection time is not in the data). Use it consistently and state this.
- With one movie per concentration, the independent unit is the experiment (n = 6). The LOCO splits give 6 paired comparisons.
- `T1_temporal_early_late` uses the same movie for training and testing. Report it only as a secondary analysis.
- The `coverage_proxy` column is a QC number (fraction of pixels > 1 nm), not the scientific coverage target. That target is built in Portion 2.
