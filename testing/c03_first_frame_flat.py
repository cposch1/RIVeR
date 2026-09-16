#!/usr/bin/env python3

from pathlib import Path
import shutil
import argparse
from PIL import Image, ImageDraw, ImageFont


def add_label(
    image_path: Path,
    camera_name: str,
    date_str: str,
    time_str: str
) -> None:
    """
    Add a top-center label containing:

    camera_name
    YYYY-MM-DD
    HH:MM:SS UTC
    """

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Format date
    if len(date_str) == 8:
        date_fmt = (
            f"{date_str[:4]}-"
            f"{date_str[4:6]}-"
            f"{date_str[6:]}"
        )
    else:
        date_fmt = date_str

    # Format time
    if len(time_str) == 6:
        time_fmt = (
            f"{time_str[:2]}:"
            f"{time_str[2:4]}:"
            f"{time_str[4:6]} UTC"
        )
    else:
        time_fmt = f"{time_str} UTC"

    text = f"{camera_name}\n{date_fmt}\n{time_fmt}"

    # Font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    # Text dimensions
    bbox = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        align="center"
    )

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding_x = 25
    padding_y = 15

    rect_w = text_w + 2 * padding_x
    rect_h = text_h + 2 * padding_y

    x = (img.width - rect_w) // 2
    y = 0

    # Black rectangle
    draw.rectangle(
        [x, y, x + rect_w, y + rect_h],
        fill="black"
    )

    # White text
    draw.multiline_text(
        (x + padding_x, y + padding_y),
        text,
        font=font,
        fill="white",
        align="center"
    )

    img.save(image_path)


def collect_frames(input_dir: Path) -> None:
    """
    Expected input structure:

    camera_name/
    ├── YYYYMMDD/
    │   └── HHMMSS/
    │       └── *.jpg
    └── ...

    Output:

    camera_name_first_frames/
    ├── camera_name_YYYYMMDD_HHMMSS.jpg
    └── ...
    """

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    camera_name = input_dir.name

    output_dir = input_dir.parent / f"{camera_name}_first_frames"
    output_dir.mkdir(exist_ok=True)

    copied = 0

    for date_dir in sorted(input_dir.iterdir()):

        if not date_dir.is_dir():
            continue

        date_str = date_dir.name

        for time_dir in sorted(date_dir.iterdir()):

            if not time_dir.is_dir():
                continue

            time_str = time_dir.name

            jpg_files = []

            for ext in (
                "*.jpg",
                "*.JPG",
                "*.jpeg",
                "*.JPEG"
            ):
                jpg_files.extend(time_dir.glob(ext))

            if not jpg_files:
                print(
                    f"WARNING: No JPG found in {time_dir}"
                )
                continue

            src_file = sorted(jpg_files)[0]

            dst_file = output_dir / (
                f"{camera_name}_{date_str}_{time_str}.jpg"
            )

            shutil.copy2(src_file, dst_file)

            add_label(
                dst_file,
                camera_name,
                date_str,
                time_str
            )

            copied += 1

            print(
                f"Copied {src_file.name} -> {dst_file.name}"
            )

    print("\nDone.")
    print(f"Copied {copied} image(s)")
    print(f"Output folder: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Flatten a camera/date/time folder structure "
            "into a single output folder. Files are "
            "renamed to "
            "'camera_name_date_time.jpg' and labeled."
        ),
        epilog=(
            "Example:\n"
            "  python c03_first_frame_flat.py "
            "C:/Users/cposch1/Desktop/ilh-cam1-pt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_directory",
        type=Path,
        help="Camera directory containing date folders."
    )

    args = parser.parse_args()

    collect_frames(args.input_directory)


if __name__ == "__main__":
    main()