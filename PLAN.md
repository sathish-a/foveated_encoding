# Phase 2 — Foveated Encoding Architecture Plan

## 2.1 Architecture

### Injection Point

**Primary injection:** `x265_picture.quantOffsets` (existing field, x265.h:475).

No modification to the x265 data path is needed. The array is already:
- Allocated and copied in `encoder.cpp:1779–1786`
- Applied additively to AQ offsets in `slicetype.cpp:634–636`
- Averaged per-CU and used in `analysis.cpp:4316–4400`

**What we add (new code only):**
1. CLI flags in `param.cpp` and `x265cli.h`/`x265cli.cpp`
2. Three new fields in `x265_param` (in `x265.h`, end of struct) for the fovea parameters
3. Frame-level logic in `x265.cpp` (the CLI application) to compute and populate `quantOffsets`
4. A separate standalone `gaze_map.c` / `gaze_map.py` for the Gaussian generator
5. A `saliency_gaze.py` preprocessing script

### Data Structure

New fields appended to `x265_param` (added at the end to maintain ABI compatibility):

```c
/* Foveated encoding parameters. All zero = disabled (no behavioral change). */
float    foveaGazeX;       /* Gaze X in pixels, 0 = left edge */
float    foveaGazeY;       /* Gaze Y in pixels, 0 = top edge  */
float    foveaDelta;       /* Max QP offset in peripheral region (0 = disabled) */
float    foveaSigma;       /* Gaussian sigma in pixels */
char*    foveaGazeFile;    /* Path to per-frame gaze file; NULL = use static gaze */
```

The `quantOffsets` array is computed on the CPU in the application layer before each `x265_encoder_encode()` call. It is not computed inside the encoder library — this keeps x265 library changes minimal.

### quantOffsets Array

- Size: `ceil(W/16) × ceil(H/16)` floats (8160 for 1920×1080)
- Layout: row-major, row=0 is top of frame
- Values: positive = higher QP = lower quality (peripheral)
- Values: zero or negative = lower QP = higher quality (foveal)
- Clamped: final QP is clamped by x265 to [qpMin, qpMax]; we pre-clamp at offset level to [−10, +delta]

**AQ requirement:** `--aq-mode` must be ≥ 1. We recommend `--aq-mode 2` (auto-variance). Our offsets are additive on top of AQ; if AQ is disabled, offsets still work (slicetype.cpp:488–494).

### Lookahead Interaction

- quantOffsets are copied in `encoder.cpp:1786` at frame submission time
- The lookahead thread reads them in `slicetype.cpp:474` and sums them with AQ variance
- cuTree (if enabled) further modifies `qpCuTreeOffset` after AQ; our offset survives in `qpAqOffset`
- **Recommendation for testing:** use `--no-cutree` initially to isolate foveal effect; re-enable cuTree later

---

## 2.2 Gaussian QP Map

### Formula

```
q(x, y) = delta × (1 − exp(−((x − x0)² + (y − y0)²) / (2σ²)))
```

Where:
- `(x0, y0)` = gaze fixation point in pixel coordinates
- `σ` = Gaussian sigma in pixels (see calculation below)
- `delta` = max QP offset in peripheral region
- Output range: [0, delta] → added to base QP (raises peripheral QP)

At the gaze center: `q = 0` (no penalty, full quality)  
At infinity: `q → delta` (max quality reduction)

### Sigma Calculation for 2.5° Visual Angle

Assumptions:
- Viewing distance D = 60 cm
- Monitor: 24 inch diagonal, 16:9 → 53.1 × 29.9 cm physical size
- Resolution: 1920 × 1080 px

Physical pixel size: 53.1 cm / 1920 px = 0.02766 cm/px

At D = 60 cm, 1° visual angle = 2 × 60 × tan(0.5°) = 1.0472 cm

Pixels per degree: 1.0472 cm / 0.02766 cm/px = **37.9 px/degree**

Sigma for 2.5° foveal radius: **2.5 × 37.9 = 94.7 ≈ 95 pixels**

For the quantOffsets 16×16 grid: sigma_blocks = 95 / 16 ≈ **5.9 blocks**

### Downsampling to QG Grid

The Gaussian is computed directly on the QG (16×16 block) grid using block center coordinates:

```python
def compute_qp_map(W, H, ctu_size=16, gaze_x=None, gaze_y=None,
                   sigma=95.0, delta=20.0):
    if gaze_x is None: gaze_x = W / 2
    if gaze_y is None: gaze_y = H / 2

    blocks_x = math.ceil(W / ctu_size)
    blocks_y = math.ceil(H / ctu_size)

    # Block center coordinates in pixels
    cx = (np.arange(blocks_x) + 0.5) * ctu_size
    cy = (np.arange(blocks_y) + 0.5) * ctu_size

    bx, by = np.meshgrid(cx, cy)  # shape: (blocks_y, blocks_x)
    dist_sq = (bx - gaze_x)**2 + (by - gaze_y)**2
    qp_map = delta * (1.0 - np.exp(-dist_sq / (2 * sigma**2)))
    return qp_map.astype(np.float32)
```

### Value Clamping Strategy

- Clamp offset to `[−delta_max, +delta_max]` where `delta_max = 40` (per spec)
- x265 already clips final QP to `[param->rc.qpMin, param->rc.qpMax]` in `calculateQpforCuSize`
- No additional clamping needed; values outside [0, 51] are safe to pass
- Delta sweep: [5, 10, 15, 20, 25, 30, 40]

---

## 2.3 Gaze Simulation Modes

### Mode 1: STATIC
- Fixed gaze at frame center: `(W/2, H/2)`
- Same `quantOffsets` array reused for all frames
- CLI: `--fovea-gaze W/2,H/2 --fovea-delta 20 --fovea-sigma 95`

### Mode 2: SALIENCY
- Preprocessing: `saliency_gaze.py` reads the YUV, computes per-frame saliency peak
- Saliency method: **Gaussian center-bias + motion energy** (no exotic deps)
  1. Convert each frame to grayscale
  2. Compute frame difference with previous frame (motion energy)
  3. Apply Gaussian center-prior map (peak at center, falloff toward edges)
  4. Saliency = motion_energy × center_prior + epsilon
  5. Gaze point = argmax of saliency map
- Output: `gaze.txt`, one line per frame: `frame_num x y`
- CLI: `--fovea-gaze-file gaze.txt --fovea-delta 20 --fovea-sigma 95`

### Mode 3: MOUSE (stretch goal)
- x265cli reads mouse position via X11/Xlib at encode time
- Lower priority; not in initial implementation

---

## 2.4 Saccade Handling

### Detection

Between consecutive frames, compare gaze points:
```python
dist = sqrt((x1 - x0)^2 + (y1 - y0)^2)
is_saccade = dist > 0.05 * frame_width   # 5% of frame width threshold
```

For 1920px: threshold = 96px (≈ 2.5°), matching sigma radius.

### Response: Selective Intra Refresh via QP Bias

On saccade detection (new gaze point at `(x1, y1)`):
1. Set `quantOffsets` at new gaze location to `−delta` (super high quality = intra preferred)
2. Set `quantOffsets` at old gaze location to `+delta` (low quality, decoder must not rely on)
3. x265's AQ + rate control will naturally prefer intra decisions in regions with very low lambda (negative QP offset)

**Why not force intra directly:** The PIR mechanism (`bIntraRefresh`) only works on column bands. Forcing specific CTUs to intra would require modifying `compressCTU()` in `analysis.cpp`. This is doable (adding a `m_forceIntraMap` uint8_t array to `Frame`) but is a larger x265 change.

**Implementation plan (two variants):**

**Variant A (no x265 modification):** Set `quantOffsets[saccade_dest_CTU] = -40.0`. With such a low lambda, x265 will use intra prediction for those CTUs naturally. Not guaranteed but usually effective.

**Variant B (surgical x265 modification):** Add `uint8_t* m_forceIntraMap` to `Frame` (frame.h:108 area). In `analysis.cpp:compressCTU()` after line 444, add:
```cpp
if (m_frame->m_forceIntraMap && m_frame->m_forceIntraMap[ctu.m_cuAddr])
    compressIntraCU(ctu, cuGeom, qp);
else
    // existing code
```

We implement Variant A first; add Variant B only if evaluation shows insufficient foveal refresh.

---

## 2.5 Evaluation Plan

### Metrics Per Encode

For each (video, delta, gaze_mode) combination:
1. **Bitrate (kbps):** from x265 encode log
2. **Global PSNR-Y:** average over all frames (from x265 `--psnr` flag)
3. **Foveal PSNR-Y:** decode output, compute PSNR for center 25% of frame area (960×540 for 1080p)
4. **Peripheral PSNR-Y:** PSNR for pixels outside center 50% of frame area
5. **Global SSIM:** from x265 `--ssim` flag
6. **Bitrate savings %:** `(baseline_bitrate − foveal_bitrate) / baseline_bitrate × 100`

### BD-Rate Computation

BD-rate computed across 7 delta values [5, 10, 15, 20, 25, 30, 40] at fixed CRF 28:
- Each delta gives a point in (bitrate, PSNR) space
- BD-rate = average bitrate difference over PSNR range (Bjontegaard delta)
- Compare against 63.24% savings from Wiedemann et al. 2020

### Encode Matrix

For each of 4 test videos:
- Baseline encode (no foveation): CRF 28
- Static center gaze, delta ∈ [5, 10, 15, 20, 25, 30, 40]
- Saliency gaze, delta ∈ [5, 10, 15, 20, 25, 30, 40]
Total: 4 × (1 + 7 + 7) = **60 encodes**

---

## 2.6 Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | AQ mode required but not enabled — quantOffsets silently ignored | High | High | Always pass `--aq-mode 2` in all encodes; add assertion check in CLI wrapper |
| 2 | cuTree modifies foveal QP offsets, reducing peripheral savings | Medium | Medium | Test with `--no-cutree` first; verify foveal/peripheral PSNR delta; if cuTree corrupts foveal bias, disable it for foveated mode |
| 3 | quantOffsets array size mismatch (qg-size vs frame size) | Medium | High | Always compute array size as `ceil(W/16)*ceil(H/16)` matching default qg-size=16; document and enforce in CLI |
| 4 | x265 compile failure due to added param fields breaking ABI | Low | High | Append new fields only at END of x265_param struct; don't insert in middle; test with `sizeof_param` check at runtime |
| 5 | Foveal PSNR degradation (injection point affects foveal region too) | Low | High | Verify center offset = 0.0 exactly; if analysis.cpp averaging introduces bleed, switch to finer-grained injection or reduce sigma |

---

## 2.7 Implementation File Map

| File | Change Type | Description |
|------|-------------|-------------|
| `source/x265.h` | ADD | 5 new fields at end of `x265_param` struct |
| `source/x265cli.h` | ADD | CLI long-option entries for fovea flags |
| `source/x265cli.cpp` | ADD | Help text for fovea options |
| `source/common/param.cpp` | ADD | OPT() parse entries for fovea params + validation |
| `source/x265.cpp` (CLI app) | ADD | Per-frame quantOffsets allocation, Gaussian computation, gaze-file reader |
| `gaze_map.py` (new, standalone) | NEW | Standalone Gaussian QP map generator + heatmap visualizer |
| `saliency_gaze.py` (new) | NEW | Per-frame saliency peak extractor, outputs gaze.txt |
| `evaluate.py` (new) | NEW | Automated evaluation across all encode combinations |

x265 library files (`encoder/*.cpp`, `common/frame.cpp`) require **zero modification** for the basic injection path. The only library change is the 5 new param fields.
