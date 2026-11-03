"""Composite the 'char' and 'shadow' passes into final white-background frames.

Usage:
    python3 composite_iris.py <char_dir> <shadow_dir> <out_dir>

For every ####.png in char_dir:
    shadow alpha (small render) is upscaled to full size with a smooth filter,
    background = white * (1 - shadow), then the mascot (straight-alpha RGBA)
    is laid over it. Writes 8-bit RGB PNGs.
An output is (re)made when it is missing or older than either input. A char
frame without its shadow frame is an error (exit code 1).
"""
import glob
import os
import sys

import cv2
import numpy as np


def srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


SHADOW_FLOOR = 0.015


def composite(char_png, shadow_png, out_png):
    ch = cv2.imread(char_png, cv2.IMREAD_UNCHANGED)
    H, W = ch.shape[:2]
    rgb = ch[..., 2::-1].astype(np.float32) / 255.0
    a = ch[..., 3:4].astype(np.float32) / 255.0
    sh = cv2.imread(shadow_png, cv2.IMREAD_UNCHANGED)
    s = sh[..., 3].astype(np.float32) / 255.0
    s = cv2.resize(s, (W, H), interpolation=cv2.INTER_CUBIC)
    s = cv2.GaussianBlur(s, (0, 0), 5.0)          # soft shadows: hide upscale steps + grain
    # drop the faint sky-occlusion haze far from the mascot so open backdrop is
    # pure white; real shadows (alpha well above this) keep their full strength
    s = np.clip((s - SHADOW_FLOOR) / (1.0 - SHADOW_FLOOR), 0, 1)[..., None]
    # mix in linear light, like Blender's Alpha Over in the compositor
    bg = 1.0 - s                                   # white backdrop, darkened by shadow
    out = srgb_to_lin(rgb) * a + bg * (1 - a)
    out = (lin_to_srgb(out) * 255.0 + 0.5).astype(np.uint8)
    tmp = out_png + ".tmp.png"
    cv2.imwrite(tmp, out[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    os.replace(tmp, out_png)


def main(char_dir, shadow_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n, missing = 0, []
    for cp in sorted(glob.glob(os.path.join(char_dir, "[0-9][0-9][0-9][0-9].png"))):
        name = os.path.basename(cp)
        sp = os.path.join(shadow_dir, name)
        op = os.path.join(out_dir, name)
        if not os.path.exists(sp):
            missing.append(name)
            continue
        if os.path.exists(op) and os.path.getmtime(op) >= max(os.path.getmtime(cp), os.path.getmtime(sp)):
            continue
        composite(cp, sp, op)
        n += 1
    print(f"composited {n} frames into {out_dir}")
    if missing:
        print(f"ERROR: no shadow frame for {len(missing)} char frames: {missing[:10]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
