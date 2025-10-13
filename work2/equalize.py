import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for file output
import matplotlib.pyplot as plt


def load_image_as_grayscale_uint8(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    if img.mode != "L":
        img = img.convert("L")
    arr = np.array(img, dtype=np.uint8)
    return arr


def save_image(array_uint8: np.ndarray, path: str) -> None:
    Image.fromarray(array_uint8, mode="L").save(path)


def compute_histogram_and_cdf(arr_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, _ = np.histogram(arr_uint8.flatten(), bins=256, range=(0, 255))
    cdf = counts.cumsum()
    return counts, cdf


def equalize_histogram(arr_uint8: np.ndarray) -> np.ndarray:
    counts, cdf = compute_histogram_and_cdf(arr_uint8)
    cdf_min = cdf[np.nonzero(cdf)][0]
    num_pixels = arr_uint8.size
    # Mapping: s_k = round((cdf(k) - cdf_min) / (num_pixels - cdf_min) * 255)
    denom = max(num_pixels - cdf_min, 1)
    cdf_norm = np.round((cdf - cdf_min) / denom * 255.0).astype(np.uint8)
    # Build LUT and apply
    lut = cdf_norm
    out = lut[arr_uint8]
    return out


def label_image(img: Image.Image, label: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    # Try to load a TTF font for robust sizing; fallback to default
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

    pad = 4
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            text_w, text_h = draw.textsize(label, font=font)
        except Exception:
            text_w = max(1, len(label) * 6)
            text_h = 11

    rect = (0, 0, text_w + pad * 2, text_h + pad * 2)
    draw.rectangle(rect, fill=0)
    draw.text((pad, pad), label, fill=255, font=font)
    return img


def plot_hist(counts: np.ndarray, path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(256), counts, width=1.0, color='#4e79a7')
    ax.set_xlim(0, 255)
    ax.set_xlabel('Gray level')
    ax.set_ylabel('Count')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cdf(cdf: np.ndarray, path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(256)
    ax.plot(x, cdf, color='#59a14f')
    ax.set_xlim(0, 255)
    ax.set_xlabel('Gray level')
    ax.set_ylabel('Cumulative count')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    base_dir = os.path.abspath('.')
    input_path = os.path.join(base_dir, 'origin.png')
    outdir = os.path.join(base_dir, 'output_eq')
    os.makedirs(outdir, exist_ok=True)

    arr_orig = load_image_as_grayscale_uint8(input_path)
    arr_eq = equalize_histogram(arr_orig)

    # Save labeled images (English overlays)
    orig_img = label_image(Image.fromarray(arr_orig), 'Original')
    eq_img = label_image(Image.fromarray(arr_eq), 'Equalized')
    img_orig_path = os.path.join(outdir, 'original.png')
    img_eq_path = os.path.join(outdir, 'equalized.png')
    orig_img.save(img_orig_path)
    eq_img.save(img_eq_path)

    # Histograms and CDFs
    h0, c0 = compute_histogram_and_cdf(arr_orig)
    h1, c1 = compute_histogram_and_cdf(arr_eq)
    hist_orig_path = os.path.join(outdir, 'hist_original.png')
    hist_eq_path = os.path.join(outdir, 'hist_equalized.png')
    cdf_orig_path = os.path.join(outdir, 'cdf_original.png')
    cdf_eq_path = os.path.join(outdir, 'cdf_equalized.png')
    plot_hist(h0, hist_orig_path, 'Original histogram')
    plot_hist(h1, hist_eq_path, 'Equalized histogram')
    plot_cdf(c0, cdf_orig_path, 'Original CDF')
    plot_cdf(c1, cdf_eq_path, 'Equalized CDF')

    # Report
    files = {
        'img_orig': img_orig_path,
        'img_eq': img_eq_path,
        'hist_orig': hist_orig_path,
        'hist_eq': hist_eq_path,
        'cdf_orig': cdf_orig_path,
        'cdf_eq': cdf_eq_path,
    }

    # Console summary
    print('Saved to:', outdir)
    for k, v in files.items():
        print(f' - {k}: {v}')


if __name__ == '__main__':
    main()


