# `fast_sharp` — run parameters

Reference record for the orthophoto at `outputs/fast_sharp/survey_orthophoto.tif`.
Everything below is what actually produced that file, including the values the engine
derived for itself.

**Source flight:** `F7/DJI_0527.MOV` + `DJI_0527.SRT` — 2024-11-03 11:35:42 IST,
27.6 s, 3840×2160 @ 23.976 fps, Tirupati AP (13.398 N, 79.564 E).

---

## 1. Frame extraction

```bash
python tools/extract_geotagged.py \
  --video D:/ortho_generator/F7/DJI_0527.MOV \
  --srt   D:/ortho_generator/F7/DJI_0527.SRT \
  --interval 1.0 \
  --out   outputs/DJI_0527_1fps
```
*(run from `api_video_ortho/`)*

| Parameter | Value | Note |
| --- | --- | --- |
| `--interval` | `1.0` s | 1 fps → **28 frames** from 27.6 s |
| Output format | JPEG q100, 4:4:4 | full native 3840×2160, no downscale |
| Attitude source | `outputs/DJI_0527_attitude.csv` | auto-discovered; see §2 |

Each frame carries EXIF (GPS, exposure, `FocalLengthIn35mmFilm`) **and** an XMP packet
with `drone-dji:GimbalYaw/Pitch/RollDegree`, `AbsoluteAltitude`, `RelativeAltitude`.
The XMP is what the ortho engine reads — EXIF has no field for gimbal pitch or roll.

## 2. Attitude recovery

The SRT contains **no** `gb_yaw` / `gb_pitch` / `gb_roll` — 0 of 661 records. Orientation
was recovered from the video itself.

```bash
python tools/derive_attitude.py \
  --video D:/ortho_generator/F7/DJI_0527.MOV \
  --srt   D:/ortho_generator/F7/DJI_0527.SRT \
  --out   outputs/DJI_0527_attitude.csv \
  --report outputs/DJI_0527_attitude_report.json
```

| Setting | Value |
| --- | --- |
| `--stride` / `--gap` / `--scale` | `4` / `8` frames / `0.5` (defaults) |
| Focal length | `28.0` mm 35 mm-equivalent, from SRT `focal_len: 280` |

Recovered, 134 homography samples (median 2602 inliers), 8 rejected by MAD:

| Quantity | Value |
| --- | --- |
| Camera pitch | **−38.11° ± 0.99** |
| Camera roll | **+1.50°** (−0.83 … +3.62 after outlier rejection) |
| Camera yaw | **153.53° ± 1.05** |
| Height above ground | **54.6 m ± 2.1** |
| Track course over ground | 160.5° (0.64° scatter in cruise) |
| Camera yaw − track bearing | −6.90° ± 0.90 |

Cross-check: `AbsoluteAltitude − RelativeAltitude` = **159.9 ± 2.0 m MSL** across the whole
200 m transect, consistent with local terrain. Confirms the SRT `altitude` field is MSL,
not AGL.

## 3. Orthophoto

```bash
python client_demo.py --engine fast --gsd 5.0 \
  --frames D:/ortho_generator/api_video_ortho/outputs/DJI_0527_1fps/frames \
  --output D:/ortho_generator/api_ortho_generator/outputs/fast_sharp
```
*(run from `api_ortho_generator/`)*

### Explicit parameters

| Parameter | Value |
| --- | --- |
| `engine` | `fast` (planar; ground-plane homography per frame) |
| `target_gsd_cm` | `5.0` |
| `blend_mode` | `multiband` — selects the best-frame compositor |
| `crs` | `AUTO` → resolved to **EPSG:32644** (WGS 84 / UTM 44N) |
| `compression` | `DEFLATE` |
| `generate_dsm` | `true` (no DSM emitted — the fast engine has no surface model) |
| `max_images` | unset (all 28) |

### Derived by the engine

These are computed at runtime, not supplied. Recorded because they determine the output.

| Quantity | Value | Where from |
| --- | --- | --- |
| Frames with usable pose | 28 / 28, all oblique | XMP gimbal tags |
| Median AGL | 55.2 m | XMP `RelativeAltitude` |
| Focal length in pixels | 2987 px @ 3840 wide | 28 mm equiv, 36 mm sensor convention |
| Nadir-equivalent GSD | 1.89 cm/px | AGL ÷ focal px |
| **Usable range factor** | **1.48 × height = 82 m** | `usable_range_factor()`, §5 |
| Canvas | 3105 × 5079 px | extent ÷ GSD |
| Mapped area | 2.41 ha (61.5 % of canvas) | alpha band |
| Exposure gain spread | 1.488 over 41 seam pairs | `_solve_gains()` |
| Seam routing | graph cut, 28 frames | `_route_seams()`, §4a |
| Runtime | 123 s | 28 frames, three passes |

### Engine constants

Defined on `OrthoMosaicEngine`; change these to trade sharpness against seam visibility.

| Constant | Value | Effect |
| --- | --- | --- |
| `SEAM_FEATHER_M` | `0.6` m | Cross-fade width at ownership seams. Wider hides seams but re-introduces averaging blur on anything with height. |
| `SEAM_SCALE` | `0.4` | Resolution at which seams are routed by graph cut. Labels are median-filtered on upscale, otherwise the boundary shows stair-steps `1/SEAM_SCALE` px wide. |
| `EDGE_EXCLUDE_PX` | `4` px | Rim of each frame barred from owning pixels; without it, interpolation against the black warp border draws a dark line along every seam. |
| `_solve_gains(clamp=…)` | `(0.82, 1.22)` | Bound on per-frame exposure gain. Both ends were reached on this clip — auto-exposure moved 1/640 → 1/500 mid-flight. |
| `NEIGHBOURS` | `2` | Frames compared for exposure, in capture order. Only nearby frames share a seam on a single transect. |

## 4. Seam routing

Nadir-weight argmax alone decides ownership on geometry only, so it hands over wherever
two frames trade nadir advantage — frequently straight across a rooftop. Because a planar
mosaic misplaces anything with height, the two frames put that roof metres apart, and the
building arrives severed and doubled.

Measured directly: after **perfect** flat-plane rectification, two adjacent frames still
place the same scene content up to **6.03 m (120 px) apart**, median 1.66 m. That is
parallax from real 3D structure, not a registration error.

`cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")` re-cuts the boundaries, pricing each
one by how much the two frames disagree along it. Seams are pushed onto ground the frames
agree on and routed around structures, so each building is served whole by a single frame.
Falls back to Voronoi seams, then to the raw argmax partition, if the cut fails.

This does **not** remove lean — see §6. It removes the *slicing*.

> No 2D stitching library can do better here. OpenCV `Stitcher`, Hugin and similar fit one
> 2D transform per image, valid for rotation-only or planar scenes. With a translating
> camera over 3D structure there is no single transform that maps ground and rooftop
> correctly at once — that is what parallax means.

## 5. Why the derived cutoff is 1.48×

Looking out at incidence angle θ, ground range is `H·tan θ` and along-range sample
spacing stretches as `(H/f)·sin θ / cos² θ`. Setting that equal to the target GSD gives the
range past which a frame can no longer honour the requested resolution:

```
k = target_gsd / nadir_gsd = 5.00 / 1.89 = 2.65
sin θ / cos² θ = k          →  θ ≈ 56°
range = H · tan θ           →  1.48 × height = 82 m
```

So the cutoff follows the requested GSD. Ask for a finer GSD and the usable range shrinks
(sharper, less coverage); ask for coarser and it grows.

## 6. Known limits of this output

- **Not a true orthophoto.** Every pixel is projected onto one flat plane at median flying
  height. Anything with height is displaced by `h·tan θ` — 89 px for a single-storey roof,
  306 px for a mature palm, at 5 cm/px. Best-frame selection makes buildings lean *crisply*
  instead of dissolving, and graph-cut seams stop them being sliced, but **the lean is
  still there**. Removing it needs a DSM, i.e. the `accurate` engine. Buildings will not
  sit square over their footprints until then.
- **Faint radiometric banding at seams** — lens vignetting. One scalar gain per frame
  cannot flatten a falloff that varies across the frame.
- **Coverage is a fan, not a strip**, because a −38° forward-looking camera images ahead of
  the aircraft. This clip was not flown as a mapping mission; a nadir gimbal and a lawnmower
  pattern would give even coverage.
- Pitch carries roughly **±2° systematic uncertainty** from focal-length coupling
  (≈ 1° of pitch per 1 mm of assumed 35 mm-equivalent focal length).

## 7. Output files

| File | Contents |
| --- | --- |
| `outputs/fast_sharp/survey_orthophoto.tif` | RGBA GeoTIFF, EPSG:32644, 5 cm/px, DEFLATE, 23.9 MB |
| `outputs/fast_sharp/survey_orthophoto_preview.png` | Web preview |

Bounds (UTM 44N): `344448.14, 1481438.81 → 344603.39, 1481692.76`
Bounds (WGS-84): `79.563387, 13.396528 → 79.564807, 13.398832`

Earlier runs kept for comparison: `fast_baseline` (original nadir similarity),
`fast_ortho` (rectified but averaged — the blurry one).
