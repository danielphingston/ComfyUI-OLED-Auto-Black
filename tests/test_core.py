import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oled_core import process  # noqa: E402


def make_scene(seed=0, h=512, w=288):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 15 / 255, np.float32)
    img += rng.normal(0, 2 / 255, (h, w, 1)).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    d = np.hypot(yy - 380, xx - 144)
    cov = np.clip((60 - d) + 0.5, 0, 1)[..., None]
    img = img * (1 - cov) + cov * np.array([0.8, 0.6, 0.5], np.float32)
    wisp = np.zeros((h, w), bool)
    for y in range(20, 250):
        x = int(80 + 30 * np.sin(y / 25))
        wisp[y, x:x + 2] = True
    img[wisp] += 11 / 255
    img[100:115, 220:235] = [0.25, 0.02, 0.02]          # deep red object
    return np.clip(img, 0, 1), wisp, d


def test_background_becomes_true_black():
    img, _, d = make_scene()
    out, mask, *_ = process(img)
    empty = np.zeros(img.shape[:2], bool)
    empty[300:330, 0:40] = True
    assert (out[empty] == 0).mean() > 0.99


def test_faint_wisps_survive():
    img, wisp, _ = make_scene()
    out, *_ = process(img)
    assert (out.max(axis=2)[wisp] * 255 > 4).mean() > 0.8


def test_detail_protection_off_loses_more_wisps():
    img, wisp, _ = make_scene()
    on = (process(img)[0].max(axis=2)[wisp] * 255 > 4).mean()
    off = (process(img, detail_protection=0)[0].max(axis=2)[wisp] * 255 > 4).mean()
    assert on > off


def test_deep_red_kept_with_protect_color():
    img, *_ = make_scene()
    out, *_ = process(img, protect_color=True)
    assert out[107, 227, 0] > 0.15


def test_subject_untouched():
    img, _, d = make_scene()
    out, *_ = process(img, shadow_dehaze=0.0, dither=False)
    core = d < 50
    assert np.abs(out[core] - img[core]).max() < 1e-4


def test_bright_image_fallback_is_gentle():
    rng = np.random.default_rng(1)
    img = (rng.random((128, 128, 3)) * 0.5 + 0.4).astype(np.float32)
    out, *_, report, bp = process(img)
    assert "fallback" in report and bp <= 2
    assert np.abs(out - img).max() < 1e-4


def test_output_shapes_and_range():
    img, *_ = make_scene()
    out, mask, dbg, report, bp = process(img)
    assert out.shape == img.shape and dbg.shape == img.shape
    assert mask.shape == img.shape[:2]
    assert out.min() >= 0 and out.max() <= 1
    assert isinstance(bp, int) and "black_point" in report


def test_no_halo_around_bright_subject():
    img, _, d = make_scene()
    out, *_ = process(img)
    ring = (d > 64) & (d < 75)          # haze right next to the disc
    assert (out.max(axis=2)[ring] == 0).mean() > 0.95


def test_soft_glow_kept_but_background_black():
    rng = np.random.default_rng(3)
    h, w = 512, 288
    yy, xx = np.mgrid[:h, :w]
    v = 13 + rng.normal(0, 2, (h, w))
    band = np.exp(-((yy - xx * 1.2 - 150) / 50.0) ** 2) * 9      # faint diffuse glow
    v = v + band
    img = np.repeat((np.clip(v, 0, 255) / 255)[..., None], 3, axis=2).astype(np.float32)
    on = process(img, keep_soft_glow=1.0)[0].max(axis=2) * 255
    off = process(img, keep_soft_glow=0.0)[0].max(axis=2) * 255
    core, empty = band > 7, band < 0.01
    assert on[core].mean() > 2 * off[core].mean()
    assert (on[empty] == 0).mean() > 0.97
