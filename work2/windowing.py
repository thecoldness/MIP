import argparse
import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_image_as_grayscale_array(image_path: str) -> np.ndarray:
    """Load an image and return a 2D numpy array (float64) in grayscale.

    Supports 8-bit and 16-bit grayscale or RGB(A) PNGs/JPEGs. If the image is
    multi-channel, convert to luminance using Pillow's L mode.
    """
    img = Image.open(image_path)

    # Convert to grayscale explicitly to ensure 2D array
    if img.mode not in ("L", "I;16"):
        img = img.convert("L")

    if img.mode == "I;16":
        arr = np.array(img, dtype=np.uint16)
    else:
        arr = np.array(img, dtype=np.uint8)

    return arr.astype(np.float64)


def apply_window(image_array: np.ndarray, window_width: float, window_level: float) -> np.ndarray:
    """Apply window width/level to a grayscale image array.

    The window is defined by [level - width/2, level + width/2]. Values outside
    this range are clamped, then linearly mapped to [0, 255]. Returns uint8.
    """
    if window_width <= 0:
        raise ValueError("window_width must be > 0")

    lower_bound = window_level - (window_width / 2.0)
    upper_bound = window_level + (window_width / 2.0)

    # Clamp and normalize
    clamped = np.clip(image_array, lower_bound, upper_bound)
    normalized = (clamped - lower_bound) / max(upper_bound - lower_bound, 1e-9)
    out = np.round(normalized * 255.0).astype(np.uint8)
    return out


def save_image(array_uint8: np.ndarray, path: str) -> None:
    Image.fromarray(array_uint8, mode="L").save(path)


def compute_statistics(image_array: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Return (min, max, p25, p50, p75) of the image array."""
    vmin = float(np.min(image_array))
    vmax = float(np.max(image_array))
    p25 = float(np.percentile(image_array, 25))
    p50 = float(np.percentile(image_array, 50))
    p75 = float(np.percentile(image_array, 75))
    return vmin, vmax, p25, p50, p75


def choose_default_parameters(image_array: np.ndarray) -> Tuple[List[float], float, List[float], float]:
    """Choose reasonable default WW and WL sets based on image distribution.

    Returns (ww_values, fixed_wl, wl_values, fixed_ww)
    """
    vmin, vmax, p25, p50, p75 = compute_statistics(image_array)
    dynamic_range = max(vmax - vmin, 1.0)

    # Window widths as fractions of dynamic range
    ww_values = [0.3 * dynamic_range, 0.6 * dynamic_range, 1.0 * dynamic_range]
    # Fixed window level at the median intensity
    fixed_wl = p50

    # Window levels at quartiles
    wl_values = [p25, p50, p75]
    # Fixed window width as half of the dynamic range (not too narrow/wide)
    fixed_ww = 0.5 * dynamic_range

    # Ensure minimum practical width
    ww_values = [max(w, 8.0) for w in ww_values]
    fixed_ww = max(fixed_ww, 8.0)

    return ww_values, fixed_wl, wl_values, fixed_ww


def label_image(img: Image.Image, label: str) -> Image.Image:
    """Draw a label onto the image (top-left)."""
    draw = ImageDraw.Draw(img)
    # Try to load a TrueType font first for better bbox support; fall back to default
    font = None
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ):
        try:
            font = ImageFont.truetype(path, size=14)
            break
        except Exception:
            pass
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    # Compute text bounding box with a robust fallback
    pad = 4
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            text_w, text_h = draw.textsize(label, font=font)
        except Exception:
            # Last-resort approximation
            text_w = max(1, len(label) * 6)
            text_h = 11

    rect = (0, 0, text_w + pad * 2, text_h + pad * 2)
    draw.rectangle(rect, fill=0)  # grayscale black
    draw.text((pad, pad), label, fill=255, font=font)
    return img


def build_grid(images: List[Image.Image], grid_size: Tuple[int, int], spacing: int = 8, bg: int = 20) -> Image.Image:
    """Create a grid image from a list of equally sized PIL images (grayscale)."""
    rows, cols = grid_size
    assert len(images) == rows * cols, "images count must match grid size"
    w, h = images[0].size
    grid_w = cols * w + (cols - 1) * spacing
    grid_h = rows * h + (rows - 1) * spacing
    grid = Image.new("L", (grid_w, grid_h), color=bg)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            x = c * (w + spacing)
            y = r * (h + spacing)
            grid.paste(images[idx], (x, y))
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply window width/level adjustments to an image.")
    parser.add_argument("--fixed_wl", type=float, default=None, help="Fixed WL for WW sweep (optional)")
    parser.add_argument("--fixed_ww", type=float, default=None, help="Fixed WW for WL sweep (optional)")
    args = parser.parse_args()

    base_dir = os.path.abspath('.')
    input_dir = os.path.join(base_dir , 'origin.png')
    output_dir = os.path.join(base_dir , 'output')

    os.makedirs(output_dir, exist_ok=True)

    # Load and analyze
    arr = load_image_as_grayscale_array(input_dir)
    vmin, vmax, p25, p50, p75 = compute_statistics(arr)
    ww_values, default_fixed_wl, wl_values, default_fixed_ww = choose_default_parameters(arr)

    fixed_wl = args.fixed_wl if args.fixed_wl is not None else default_fixed_wl
    fixed_ww = args.fixed_ww if args.fixed_ww is not None else default_fixed_ww

    # Normalize input image for reference output
    reference = np.round((arr - vmin) / max(vmax - vmin, 1e-9) * 255.0).astype(np.uint8)
    ref_path = os.path.join(output_dir, "reference_stretched.png")
    save_image(reference, ref_path)

    # Fixed WL, vary WW
    ww_files: List[str] = []
    ww_images_for_grid: List[Image.Image] = []
    for ww in ww_values:
        out_arr = apply_window(arr, ww, fixed_wl)
        fname = os.path.join(output_dir, f"fixedWL_{fixed_wl:.1f}_WW_{ww:.1f}.png")
        save_image(out_arr, fname)
        ww_files.append(fname)
        labeled = label_image(Image.fromarray(out_arr), f"WL {fixed_wl:.1f}, WW {ww:.1f}")
        ww_images_for_grid.append(labeled)

    # Fixed WW, vary WL
    wl_files: List[str] = []
    wl_images_for_grid: List[Image.Image] = []
    for wl in wl_values:
        out_arr = apply_window(arr, fixed_ww, wl)
        fname = os.path.join(output_dir, f"fixedWW_{fixed_ww:.1f}_WL_{wl:.1f}.png")
        save_image(out_arr, fname)
        wl_files.append(fname)
        labeled = label_image(Image.fromarray(out_arr), f"WL {wl:.1f}, WW {fixed_ww:.1f}")
        wl_images_for_grid.append(labeled)

    # Build composite grid: first row WW sweep, second row WL sweep
    # Use the smallest common size in case of any metadata differences
    target_w, target_h = ww_images_for_grid[0].size
    ww_images_for_grid = [im.resize((target_w, target_h)) for im in ww_images_for_grid]
    wl_images_for_grid = [im.resize((target_w, target_h)) for im in wl_images_for_grid]
    grid = build_grid(ww_images_for_grid + wl_images_for_grid, grid_size=(2, len(ww_images_for_grid)), spacing=12, bg=10)
    grid_path = os.path.join(output_dir, "windowing_grid.png")
    grid.convert("L").save(grid_path)

    # Console summary
    print("Reference stretched image:", ref_path)
    print("Fixed WL, varying WW images:")
    for f in ww_files:
        print(" -", f)
    print("Fixed WW, varying WL images:")
    for f in wl_files:
        print(" -", f)
    print("Composite grid:", grid_path)


if __name__ == "__main__":
    main()


