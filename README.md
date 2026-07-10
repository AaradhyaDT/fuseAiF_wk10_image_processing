# Week 10 — Image Processing (FreshTrack CV Pipeline)

**Fusemachines AI Fellowship 2026** · Instructor: Rakshya Lama Moktan · 100 pts · Due Jul 16, 20:45 NPT

Classical computer vision pipeline covering color-space segmentation, morphology, edge detection, and feature detection — built and validated against real classroom images, not synthetic placeholders.

## Stack

Python · OpenCV · NumPy · Matplotlib · kagglehub

## Dataset

- **Fruits-360** (`moltean/fruits`, 100×100 variant, via `kagglehub`) — auto-mapped to 6 required classes: `red_apple`, `green_apple`, `banana`, `strawberry`, `orange`, `lime`
- **Instructor-provided** (Classroom upload): `morphology.png` (cursive "j"), `chessboard.png`, `mixed_fruit_bowl.jpeg`

## Structure

**Setup** — Kaggle auth (Colab Secrets), dataset download, class→file resolution, instructor image upload, sanity check (9 images staged in `/content/assignment_images/`).

### Part A — Color Space & Fruit Segmentation

- Q1–2: BGR/RGB and HSV channel inspection
- Q3–10: Per-fruit `inRange()` masks
- Multi-class segmentation on `mixed_fruit_bowl.jpeg`
- Two HSV range sets, kept separate since they come from different image sources: `HSV_RANGES` (Fruits-360 studio thumbnails, 6 classes, estimated) vs `BOWL_HSV_RANGES` (real-sampled from the bowl photo — apple H≈171, banana H≈25, orange H≈16; orange/banana separated mainly by saturation, not hue)

### Part B — Morphology & Denoising

- 5 ops (erosion, dilation, opening, closing, gradient) on the real `morphology.png`
- Salt-and-pepper / Gaussian noise injection + Gaussian, median, bilateral denoising

### Part C — Edge Detection

- Q11: From-scratch Canny (Sobel → NMS → double-threshold → hysteresis) vs `cv2.Canny()`, compared via pixel-agreement metric. Pure-Python NMS/hysteresis is O(H×W) with nested loops, so it runs on a resized 150×150 copy for this comparison step only.

### Part D — Feature Detection

- Q12: Harris corners on `chessboard.png`, 5-way threshold sweep
- Q13: Hough circles on the bowl photo, tuned `param2=100`
- Q14: Full single-fruit detection pipeline (HSV mask → morphology → contour → bbox), using `BOWL_HSV_RANGES`

## Notes from the real-data pass

- `morphology.png` is a cursive letter stroke, not a noisy blob — write-up reflects actual behavior (erosion nearly erases the thin stroke; opening/closing barely change it, since there are no specks/holes to remove).
- Filename mismatch handled: notebook originally expected `mixed_bowl.jpg`, classroom file is `mixed_fruit_bowl.jpeg` — upload cell normalizes this.
- Hough params retuned from synthetic defaults, which gave 128 false positives on the real photo, down to `param2=100`.
- Q14 caught a real segmentation limitation: touching apples in the bowl initially produced one bounding box spanning all 5 (w=1060, h=466). Fixed with distance-transform + connected-components separation, verified against a single apple-sized bbox (w=286, h=315).
- `ACTUAL_FRUIT_COUNT` is left as `None` — color-mask contour counting can't reliably separate touching same-color objects, so this is filled in by manual count rather than an unverified guess.

## Setup

1. Add `KAGGLE_USERNAME` / `KAGGLE_KEY` via Colab Secrets (🔑 icon), or use the commented file-upload fallback in the setup cell.
2. Run the notebook top to bottom on a fresh runtime.

## License

See `LICENSE`(LICENSE).
