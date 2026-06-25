import argparse
from pathlib import Path
from PIL import Image


def decompose_gif(gif_path, output_dir=None, fmt="png"):
    gif_path = Path(gif_path)
    if not gif_path.exists():
        raise FileNotFoundError(f"File not found: {gif_path}")

    if output_dir is None:
        output_dir = gif_path.parent / gif_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gif = Image.open(gif_path)
    n_frames = gif.n_frames
    print(f"Found {n_frames} frames in '{gif_path.name}'")

    for i in range(n_frames):
        gif.seek(i)
        frame = gif.convert("RGBA")
        out_path = output_dir / f"frame_{i:04d}.{fmt}"
        frame.save(out_path)
        print(f"  Saved {out_path}")

    print(f"\nDone — {n_frames} frames saved to '{output_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose a GIF into individual frames.")
    parser.add_argument("gif_path", help="Path to the input GIF file")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: <gif_name>/)")
    parser.add_argument("--format", "-f", default="png", choices=["png", "jpg", "bmp"], help="Output image format (default: png)")
    args = parser.parse_args()

    decompose_gif(args.gif_path, args.output, args.format)