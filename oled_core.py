"""
Core OLED black processing (pure numpy, no torch) so it can be unit-tested
outside ComfyUI. Images are float32 arrays in [0, 1], shape (H, W, 3).

Pipeline (all decisions are made on v = max(R,G,B), then applied as a single
per-pixel gain so hue never shifts):

 1. analyze   - find the background haze level (bg) and its noise sigma from
                the dark histogram; derive black point t0 and toe end t1.
 2. structure - local std-dev vs. noise sigma. Real detail (smoke, hair,
                anti-aliased subject edges) varies much more than flat haze.
 3. tone      - flat haze gets a smooth toe to true black; structured pixels
                get a linear stretch instead, so faint detail survives with
                its contrast intact. Optional shadow dehaze removes the milky
                veil from the subject's shadows.
 4. despeckle - isolated 1-2 px dots inside black are removed; thin lines
                (wisps, hair, rim light) are kept.
 5. dither    - tiny triangular dither in the toe to avoid 8-bit banding.
"""
import numpy as np

PRESETS = {
    # k = how many noise sigmas above the haze the black point sits
    "gentle": 2.0,
    "balanced": 3.0,
    "aggressive": 5.0,
}


# ---------------------------------------------------------------- helpers
def _box_mean(m: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window via an integral image; edges clamped."""
    h, w = m.shape
    p = np.pad(m, r, mode="edge").astype(np.float64)
    ii = np.zeros((h + 2 * r + 1, w + 2 * r + 1), dtype=np.float64)
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    return (s / (k * k)).astype(np.float32)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _local_std(v255: np.ndarray, r: int) -> np.ndarray:
    m = _box_mean(v255, r)
    m2 = _box_mean(v255 * v255, r)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


# ---------------------------------------------------------------- analysis
def analyze(v: np.ndarray, k: float, max_black_point: float):
    """Returns dict with bg, noise, t0, t1 (0..255 scale) and diagnostics."""
    flat = v.ravel()
    if flat.size > 1_000_000:
        flat = flat[:: flat.size // 1_000_000 + 1]

    dark = flat[flat < 64 / 255.0]
    dark_frac = dark.size / flat.size

    if dark_frac < 0.10:
        return {"bg": 0.0, "noise": 1.0, "t0": 2.0, "t1": 5.0,
                "dark_frac": dark_frac, "mode": "bright-image fallback"}

    hist, _ = np.histogram(dark * 255.0, bins=64, range=(0, 64))
    hist_s = np.convolve(hist, [1, 2, 3, 2, 1], mode="same")
    bg = float(np.argmax(hist_s)) + 0.5

    win = dark * 255.0
    win = win[np.abs(win - bg) < 12]
    if win.size > 32:
        noise = float(np.median(np.abs(win - np.median(win)))) * 1.4826
    else:
        noise = 1.0
    noise = max(noise, 1.0)

    t0 = float(np.clip(bg + k * noise, 3.0, max_black_point))
    t1 = t0 + max(4.0, 0.6 * t0)
    return {"bg": bg, "noise": noise, "t0": t0, "t1": t1,
            "dark_frac": dark_frac, "mode": "auto"}


# ---------------------------------------------------------------- main
def process(img: np.ndarray, *, mode="auto", preset="balanced",
            manual_black_point=8, manual_range=10, max_black_point=40,
            protect_color=True, detail_protection=1.0, shadow_dehaze=0.5,
            despeckle=True, dither=True, seed=0):
    """
    Returns (out_img, black_mask, debug_img, report_str, black_point_int).
    """
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    h, w, _ = img.shape

    if protect_color:
        v = img.max(axis=2)
    else:
        v = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    v255 = v * 255.0

    if mode == "auto":
        a = analyze(v, PRESETS.get(preset, 3.0), max_black_point)
    else:
        t0 = float(manual_black_point)
        a = {"bg": t0 * 0.6, "noise": max(1.0, t0 * 0.15), "t0": t0,
             "t1": t0 + max(manual_range, 1), "mode": "manual",
             "dark_frac": float((v < 64 / 255).mean())}
    bg, noise, t0, t1 = a["bg"], a["noise"], a["t0"], a["t1"]

    # --- 2. structure map (resolution-aware radius) ---------------------
    r = max(1, round(min(h, w) / 600))
    sd = _local_std(v255, r)
    struct = _smoothstep(1.5 * noise, 3.0 * noise, sd) * float(detail_protection)

    # --- 3. tone curve, evaluated in value space -------------------------
    # shadow dehaze: pull shadows down by up to `haze`, fading out by mid-grey
    haze = bg * float(shadow_dehaze) if a["mode"] != "bright-image fallback" else 0.0
    def dehaze(x):  # x in 0..255
        return np.maximum(x - haze * (1.0 - _smoothstep(0.0, 128.0, x)), 0.0)

    toe = _smoothstep(t0, t1, v255)
    flat_out = dehaze(v255) * toe                       # crush flat haze

    lo = bg + 2.5 * noise                               # below this = haze, even if "structured"
    t1_out = float(dehaze(np.array(t1)))
    lin = t1_out * np.clip((v255 - lo) / max(t1 - lo, 1e-3), 0.0, 1.0)
    detail_out = np.where(v255 >= t1, flat_out, np.maximum(flat_out, lin))

    out_v = flat_out + (detail_out - flat_out) * struct

    # --- 4. despeckle: isolated dots only, lines survive -----------------
    speck = np.zeros_like(v, dtype=bool)
    if despeckle:
        black = (out_v < 0.5).astype(np.float32)
        rs = max(2, r)
        surround = _box_mean(black, rs)
        # a 1-px line through a 5x5 window leaves ~80% black; a dot leaves ~96%
        speck = (surround > 0.92) & (v255 < 2.0 * t1) & (out_v > 0)
        out_v = np.where(speck, 0.0, out_v)

    # --- apply as hue-preserving gain ------------------------------------
    gain = np.where(v255 > 1e-4, out_v / np.maximum(v255, 1e-4), 0.0)
    gain = np.clip(gain, 0.0, 4.0)
    out = np.clip(img * gain[..., None], 0.0, 1.0)

    # --- 5. dither the toe so it doesn't band in 8-bit ------------------
    if dither:
        zone = (out_v > 0) & (out_v < t1_out)
        if zone.any():
            rng = np.random.default_rng(seed)
            n = (rng.random((h, w), np.float32) - rng.random((h, w), np.float32)) / 255.0
            out = np.where(zone[..., None], np.clip(out + n[..., None], 1 / 255.0, 1.0), out)

    out[out_v <= 0] = 0.0                               # exact #000000

    mask = (out_v <= 0).astype(np.float32)

    # --- debug overlay: red = crushed, green = detail kept, blue = speck removed
    grey = np.repeat((v * 0.6)[..., None], 3, axis=2)
    dbg = grey.copy()
    crushed = (mask > 0) & (v255 > 0.5)
    kept = (struct > 0.5) & (v255 < t1) & (out_v > 0)
    dbg[crushed] = dbg[crushed] * 0.3 + np.array([0.6, 0.0, 0.0])
    dbg[kept] = np.array([0.1, 0.9, 0.2])
    dbg[speck] = np.array([0.2, 0.5, 1.0])

    pure = float(mask.mean()) * 100
    report = (f"[{a['mode']}] haze≈{bg:.1f} σ≈{noise:.1f} → black_point={t0:.0f} "
              f"toe_to={t1:.0f} dehaze={haze:.1f} | dark {a['dark_frac']*100:.0f}% | "
              f"pure black {pure:.1f}% | detail kept {kept.mean()*100:.2f}% | "
              f"specks {int(speck.sum())}")
    return out, mask, dbg, report, int(round(t0))
