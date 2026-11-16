"""Image-level input transforms and reversible crop geometry (no learned state)."""
from functools import lru_cache
import cv2
import numpy as np


def scalar_view(image, kind):
    """Compute BEFORE cropping: a patch never estimates its own normalization."""
    x = np.asarray(image, dtype=np.float32) / 255.
    if kind == "raw":
        return x
    if kind == "percentile":
        # Exclude black off-disk pixels; statistics use only this observed image.
        v = x[x > .05]
        lo, hi = np.percentile(v, (1, 99)) if len(v) else (0., 1.)
        return np.clip((x-lo) / max(float(hi-lo), .05), 0, 1)
    if kind == "clahe":
        return cv2.createCLAHE(clipLimit=2., tileGridSize=(16, 16)).apply(image) / np.float32(255)
    if kind in ("gamma08", "gamma12"):
        return x ** (.8 if kind == "gamma08" else 1.2)
    if kind == "log":
        return np.log1p(4*x) / np.float32(np.log(5))
    if kind in ("unsharp", "highpass", "blur", "detail"):
        blur = cv2.GaussianBlur(x, (0, 0), 8.)
        if kind == "blur":
            return blur
        if kind == "unsharp":
            return np.clip(x + .5*(x-blur), 0, 1)
        return np.clip(.5 + 2*(x-blur), 0, 1)
    raise ValueError(kind)


def input_view(image, spec):
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Expected a uint8 grayscale observation")
    kind = spec["channels"]
    if kind == "repeat":
        x = scalar_view(image, spec["transform"])
        parts = [x, x, x]
    elif kind == "raw_clahe_detail":
        parts = [scalar_view(image, k) for k in ("raw", "clahe", "detail")]
    elif kind == "raw_blur_detail":
        parts = [scalar_view(image, k) for k in ("raw", "blur", "detail")]
    elif kind == "raw_gamma_pair":
        parts = [scalar_view(image, k) for k in ("raw", "gamma08", "gamma12")]
    elif kind == "raw_clahe_radius":
        h, w = image.shape
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        radius = np.clip(np.hypot((xx+.5-w/2)/(w/2), (yy+.5-h/2)/(h/2)), 0, 1)
        parts = [scalar_view(image, k) for k in ("raw", "clahe")] + [radius]
    else:
        raise ValueError(kind)
    return np.stack(parts, axis=-1).astype(np.float32)


def starts(full, extent):
    if not 0 < extent <= full:
        raise ValueError("Invalid field of view")
    stride = max(1, extent*3//4)
    values = list(range(0, full-extent+1, stride))
    if values[-1] != full-extent:
        values.append(full-extent)
    return values


def boxes(full, extent):
    return [(x, y, x+extent, y+extent) for y in starts(full, extent) for x in starts(full, extent)]


def crop_resize(image, box, size, *, mask=False):
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    interp = cv2.INTER_NEAREST if mask else (cv2.INTER_AREA if x1-x0 >= size else cv2.INTER_LINEAR)
    return cv2.resize(crop, (size, size), interpolation=interp)


@lru_cache(maxsize=16)
def blend_window(extent):
    w = np.maximum(np.hanning(extent), .05).astype(np.float32)
    return w[:, None]*w[None, :]


def restore_tiles(tiles, tile_boxes, full):
    total = np.zeros((full, full), np.float32)
    weights = np.zeros_like(total)
    count = 0
    for tile, (x0, y0, x1, y1) in zip(tiles, tile_boxes, strict=True):
        extent = x1-x0
        if y1-y0 != extent or not np.isfinite(tile).all():
            raise ValueError("Invalid tile")
        window = blend_window(extent)
        prob = cv2.resize(tile.astype(np.float32), (extent, extent), interpolation=cv2.INTER_LINEAR)
        total[y0:y1, x0:x1] += prob*window
        weights[y0:y1, x0:x1] += window
        count += 1
    if count == 0 or np.any(weights <= 0):
        raise ValueError("Tile reconstruction has uncovered pixels")
    return total/weights
