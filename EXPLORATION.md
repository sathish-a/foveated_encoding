# Phase 1.2 — x265 Codebase Exploration Findings

## Q1: Per-CTU QP Decision Call Chain

The chain from top-level API call to per-CTU QP finalization:

```
x265_encoder_encode()                  [source/encoder/api.cpp]
  └─ Encoder::encode()                 [source/encoder/encoder.cpp:~1607]
       └─ Frame::create()              [source/common/frame.cpp:91]
            (allocates m_quantOffsets, copies from x265_picture.quantOffsets)
       └─ Lookahead::addPicture()      [source/encoder/slicetype.cpp]
            └─ Lookahead::slicetypeDecide()
                 └─ compressFrame()    [source/encoder/slicetype.cpp:~474]
                      (applies quantOffsets → m_lowres.qpAqOffset and qpCuTreeOffset)
       └─ FrameEncoder::startCompressFrame()   [source/encoder/frameencoder.cpp]
            └─ Analysis::compressCTU()         [source/encoder/analysis.cpp:~259]
                 └─ Analysis::calculateQpforCuSize()  [source/encoder/analysis.cpp:4316]
                      (reads qpAqOffset / qpCuTreeOffset, averages over CU block,
                       returns final integer QP for the CU)
                 └─ Analysis::setLambdaFromQP()
                      (applies QP to lambda for RD cost and quantization)
```

Key files and functions:
| File | Function | Role |
|------|----------|------|
| `source/encoder/encoder.cpp:1779–1786` | `Encoder::encode()` | Copies `quantOffsets` from `x265_picture` → `Frame::m_quantOffsets` |
| `source/common/frame.cpp:91` | `Frame::create()` | Allocates `m_quantOffsets` array of size = number of 16x16 blocks |
| `source/encoder/slicetype.cpp:474–638` | `compressFrame()` | In lookahead: adds quantOffsets into `m_lowres.qpAqOffset[]` and `qpCuTreeOffset[]` |
| `source/encoder/analysis.cpp:4316–4400` | `Analysis::calculateQpforCuSize()` | Final QP per CU: averages 16x16-grid qpAqOffset over the CU size, clips to [qpMin, qpMax] |
| `source/encoder/analysis.cpp:259` | `Analysis::compressCTU()` | Calls `calculateQpforCuSize()` for the CTU root, then recursively for sub-CUs |

---

## Q2: Existing Per-CTU QP Offset Mechanism

**Yes — `x265_picture.quantOffsets` already exists.**

Declared in `source/x265.h` line 475:
```c
float *quantOffsets;
```

Comment (x265.h:470–474):
> An array of quantizer offsets to be applied to this image during encoding.
> These are added on top of the decisions made by rateControl.
> **Adaptive quantization must be enabled** to use this feature.
> These quantizer offsets should be given for each **16×16 block**
> (8×8 block, when qg-size is 8).

Array size for 1920×1080 with default qg-size=16:
- Blocks per row: ceil(1920/16) = 120
- Blocks per col: ceil(1080/16) = 68
- Total: **8160 float values per frame**

**Requirement:** AQ mode must be non-zero (`--aq-mode 1` or higher) for quantOffsets to take effect.

---

## Q3: Lookahead Interaction

The lookahead in `slicetype.cpp:compressFrame()` reads `quantOffsets` and writes them into two arrays on the lowres frame:
- `m_lowres.qpAqOffset[]` — used when cuTree is disabled or frame is not referenced
- `m_lowres.qpCuTreeOffset[]` — used when cuTree is enabled and frame is referenced

Guard at `slicetype.cpp:481`:
```cpp
if (!(param->rc.bStatRead && param->rc.cuTree && IS_REFERENCED(curFrame)))
```
This means: if cuTree is enabled **and** we are reading from a stats file (2-pass mode), the block is skipped and the precomputed stats-file data is used instead.

**Injection strategy:** Our per-frame `quantOffsets` array is applied INSIDE the lookahead analysis, before cuTree runs. The cuTree propagation step further modifies `qpCuTreeOffset` based on temporal complexity. With our foveal offsets pre-loaded, the final CU QP = base_QP + foveal_offset + cuTree_adjustment.

**Recommendation:** For a clean injection, enable `--aq-mode 2` (auto-variance) and disable cuTree (`--no-cutree`) in initial testing to avoid cuTree modifying our foveal map. Enable cuTree later and verify that the foveal bias persists.

---

## Q4: AQ-Mode Pipeline and Injection Point Relationship

AQ-mode processes the entire frame in the **lookahead** thread:

```
slicetype.cpp::compressFrame()
  if aqMode == NONE:
      → quantOffsets applied directly as qpAqOffset (if quantOffsets present)
  else (aqMode == VARIANCE or AUTO_VARIANCE):
      → per-block variance computed via acEnergyCu()
      → qp_adj = f(variance)
      → if quantOffsets: qp_adj += quantOffsets[blockXY]   ← line 635
      → qpAqOffset[blockXY] = qp_adj
```

**Our quantOffsets are summed ON TOP of the AQ variance adjustment.** This is the correct behavior: AQ handles texture/complexity-based QP variation, our foveal offsets add an additional spatial bias on top. The two are additive.

**Injection point: `x265_picture.quantOffsets` array, set before `x265_encoder_encode()` is called.**  
No source code modification needed for the data path — only CLI plumbing and the Gaussian map generator are new code.

The final QP is clipped to `[param->rc.qpMin, param->rc.qpMax]` in `calculateQpforCuSize()` (analysis.cpp:4334–4337), so values outside [0, 51] are safe to pass as offsets; they will be clamped.

---

## Q5: MV-HEVC View Separation and Per-View QP Maps

MV-HEVC (multi-view) is exposed via `x265_param.numViews` (x265.h:2359) and `numScalableLayers` (x265.h:2356). The encoder uses separate `x265_picture` objects per view. Since `quantOffsets` is a per-picture field, **different quantOffset arrays per view are already naturally supported** — just supply a different array on each view's `x265_picture`.

There is no pre-existing "per-view QP map" API beyond this. For a stereo foveated VR scenario, you would compute one gaze-point-derived quantOffsets array and pass the same array to both eye views (or slightly different ones if IPD-adjusted gaze positions differ). The architecture we're building is compatible.

---

## Q6: CTU Size and Grid Dimensions

Default CTU size: **64×64 pixels** (set in `source/common/param.cpp:185`: `param->maxCUSize = 64`).

Some presets override to 32×32 (e.g., `param.cpp:477` for certain profiles).

For **1920×1080 at default 64×64 CTU**:
- CTUs per row: ceil(1920/64) = **30**
- CTUs per col: ceil(1080/64) = **17**
- Total CTUs per frame: **510**

However, `quantOffsets` operates at **16×16 granularity** (the quantization group, QG):
- QG blocks per row: ceil(1920/16) = **120**
- QG blocks per col: ceil(1080/16) = **68**
- Total QG blocks per frame: **8160**

Each 64×64 CTU contains a 4×4 grid of 16×16 QG blocks. `calculateQpforCuSize` (analysis.cpp:4374–4383) averages over however many 16×16 blocks fall within the current CU at the given recursion depth, so the Gaussian map needs to be defined at 16×16 resolution.

---

## Forced Intra for Saccade Handling

x265 has a Progressive Intra Refresh (PIR) mechanism via `m_param->bIntraRefresh` (encoder.cpp:4090, analysis.cpp:445). It forces `compressIntraCU()` for CTUs in columns `[pirStartCol, pirEndCol)`.

For our saccade detection, we will **not** use PIR (it is column-band based, not arbitrary spatial). Instead, we will inject large positive `quantOffsets` in the **previous** frame's peripheral map to force I-mode decisions indirectly, OR we will modify the CTU analysis to call `compressIntraCU()` based on a per-CTU saccade flag we store in `Frame`. This requires a small struct extension to `Frame` — a `uint8_t* m_forceIntraMap` bitfield.

Alternatively, the cleanest approach is to set the `quantOffsets` delta very high (e.g., +51) at the saccade destination CTUs in the frame AFTER the saccade, which naturally drives those CTUs toward intra refresh via the rate control path.

---

## Summary Table

| Question | Answer |
|----------|--------|
| Injection point | `x265_picture.quantOffsets` → copied in `encoder.cpp:1786`, applied in `slicetype.cpp:635`, read in `analysis.cpp:4363` |
| Existing mechanism | Yes — `quantOffsets` field exists and is fully plumbed |
| Lookahead interaction | Additive: our offsets + AQ variance offsets in lookahead; cuTree further modifies if enabled |
| AQ-mode relationship | Our offsets are ADDED to AQ adjustments (line 635); we inject AFTER AQ |
| MV-HEVC per-view QP | Implicit per-picture: pass different `quantOffsets` on each view's `x265_picture` |
| CTU size (default) | 64×64px; `quantOffsets` at 16×16 QG = 8160 values for 1920×1080 |
| Default maxCUSize | 64 (param.cpp:185); 32 for some profiles/presets |
| Saccade forced intra | Use large QP offset (+40) at saccade destination, or add `m_forceIntraMap` to Frame |
