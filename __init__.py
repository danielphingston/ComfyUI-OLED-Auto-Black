"""
OLED Auto Black — ComfyUI custom node.
Auto-detects the background haze per image and crushes it to true #000000,
while protecting real detail (smoke, hair, rim light, anti-aliased edges).
"""
import numpy as np
import torch

from .oled_core import process, PRESETS


class OledAutoBlack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "mode": (["auto", "manual"], {"default": "auto"}),
                "preset": (list(PRESETS.keys()), {"default": "balanced",
                           "tooltip": "How far above the detected haze the black point sits."}),
                "detail_protection": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Keep faint structured detail (smoke, hair, edges) instead of crushing it. 0 = off."}),
                "shadow_dehaze": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "Remove the milky veil from the subject's shadows. Fades out by mid-grey."}),
                "despeckle": ("BOOLEAN", {"default": True,
                              "tooltip": "Remove isolated noise dots in black areas; thin lines are kept."}),
                "dither": ("BOOLEAN", {"default": True,
                           "tooltip": "Tiny dither in the fade-to-black to prevent 8-bit banding. Turn off before an upscaler, on for the final save."}),
                "protect_color": ("BOOLEAN", {"default": True,
                                  "tooltip": "Judge darkness by max(R,G,B) so deep saturated colours survive."}),
                "max_black_point": ("INT", {"default": 40, "min": 4, "max": 96,
                                    "tooltip": "Safety cap for auto mode (0-255 scale)."}),
                "manual_black_point": ("INT", {"default": 8, "min": 0, "max": 128}),
                "manual_range": ("INT", {"default": 10, "min": 1, "max": 128}),
                "keep_soft_glow": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "Keep diffuse light just above the haze (Milky Way, fog, nebula, soft gradients). 0 = off, the deepest black."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "black_mask", "debug_view", "report", "black_point")
    FUNCTION = "run"
    CATEGORY = "image/postprocessing"

    def run(self, image, enabled, mode, preset, detail_protection, shadow_dehaze,
            despeckle, dither, protect_color, max_black_point,
            manual_black_point, manual_range, keep_soft_glow=0.0):
        if not enabled:
            b, h, w, _ = image.shape
            return (image, torch.zeros((b, h, w)), image, "disabled", 0)

        outs, masks, dbgs, reports, bps = [], [], [], [], []
        for i in range(image.shape[0]):
            arr = image[i, ..., :3].cpu().float().numpy()
            o, m, d, r, bp = process(
                arr, mode=mode, preset=preset,
                manual_black_point=manual_black_point, manual_range=manual_range,
                max_black_point=max_black_point, protect_color=protect_color,
                detail_protection=detail_protection, shadow_dehaze=shadow_dehaze,
                despeckle=despeckle, dither=dither,
                keep_soft_glow=keep_soft_glow, seed=i,
            )
            outs.append(torch.from_numpy(o))
            masks.append(torch.from_numpy(m))
            dbgs.append(torch.from_numpy(np.clip(d, 0, 1).astype(np.float32)))
            reports.append(f"#{i}: {r}")
            bps.append(bp)

        report = "\n".join(reports)
        print(f"[OLED Auto Black] {report}")
        return (torch.stack(outs).to(image.device), torch.stack(masks),
                torch.stack(dbgs), report, int(np.median(bps)))


NODE_CLASS_MAPPINGS = {"OledAutoBlack": OledAutoBlack}
NODE_DISPLAY_NAME_MAPPINGS = {"OledAutoBlack": "OLED Auto Black"}
