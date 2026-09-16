#!/usr/bin/env python3

from pathlib import Path
import shutil
import argparse
from datetime import datetime, timedelta

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

    date_fmt = (
        f"{date_str[:4]}-"
        f"{date_str[4:6]}-"
        f"{date_str[6:]}"
    )

    time_fmt = (
        f"{time_str[:2]}:"
        f"{time_str[2:4]}:"
        f"{time_str[4:6]} UTC"
    )

    text = f"{camera_name}\n{date_fmt}\n{time_fmt}"

    try:
        font = ImageFont.truetype(
            "DejaVuSans-Bold.ttf",
            20
        )
    except Exception:
        font = ImageFont.load_default()

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

    rect_w = text_w + (2 * padding_x)
    rect_h = text_h + (2 * padding_y)

    x = (img.width - rect_w) // 2
    y = 0

    draw.rectangle(
        [x, y, x + rect_w, y + rect_h],
        fill="black"
    )

    draw.multiline_text(
        (x + padding_x, y + padding_y),
        text,
        font=font,
        fill="white",
        align="center"
    )

    img.save(image_path)


def create_black_frame(
    output_file: Path,
    reference_size: tuple,
    camera_name: str,
    date_str: str,
    time_str: str,
) -> None:
    """
    Create a black image and add the label.
    """

    img = Image.new(
        "RGB",
        reference_size,
        "black"
    )

    img.save(output_file)

    add_label(
        output_file,
        camera_name,
        date_str,
        time_str,
    )


def collect_frames(
    input_dir: Path,
    full_timeseries=None,
) -> None:
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

    output_dir = (
        input_dir.parent /
        f"{camera_name}_first_frames"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    existing_images = {}

    reference_size = None

    # -----------------------------------------
    # Scan source imagery
    # -----------------------------------------

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
                "*.JPEG",
            ):
                jpg_files.extend(
                    time_dir.glob(ext)
                )

            if not jpg_files:
                continue

            src_file = sorted(jpg_files)[0]

            timestamp = datetime.strptime(
                f"{date_str}_{time_str}",
                "%Y%m%d_%H%M%S"
            )

            existing_images[timestamp] = src_file

            if reference_size is None:

                with Image.open(src_file) as img:
                    reference_size = img.size

    if not existing_images:
        raise RuntimeError(
            "No images found."
        )

    # -----------------------------------------
    # Standard mode
    # -----------------------------------------

    if full_timeseries is None:

        copied = 0

        for timestamp, src_file in sorted(
            existing_images.items()
        ):

            date_str = timestamp.strftime(
                "%Y%m%d"
            )

            time_str = timestamp.strftime(
                "%H%M%S"
            )

            dst_file = output_dir / (
                f"{camera_name}_{date_str}_{time_str}.jpg"
            )

            shutil.copy2(
                src_file,
                dst_file
            )

            add_label(
                dst_file,
                camera_name,
                date_str,
                time_str
            )

            copied += 1

            print(
                f"Copied -> {dst_file.name}"
            )

        print("\nDone.")
        print(f"Copied images : {copied}")
        print(f"Output folder : {output_dir}")

        return

    # -----------------------------------------
    # Full timeseries mode
    # -----------------------------------------

    start_date, end_date = full_timeseries

    start_dt = datetime.strptime(
        start_date,
        "%Y-%m-%d"
    )

    end_dt = datetime.strptime(
        end_date,
        "%Y-%m-%d"
    ).replace(
        hour=22,
        minute=0,
        second=0,
        microsecond=0
    )

    copied = 0
    created = 0

    current = start_dt

    while current <= end_dt:

        date_str = current.strftime(
            "%Y%m%d"
        )

        time_str = current.strftime(
            "%H%M%S"
        )

        dst_file = output_dir / (
            f"{camera_name}_{date_str}_{time_str}.jpg"
        )

        if current in existing_images:

            shutil.copy2(
                existing_images[current],
                dst_file
            )

            add_label(
                dst_file,
                camera_name,
                date_str,
                time_str
            )

            copied += 1

            print(
                f"Copied -> {dst_file.name}"
            )

        else:

            create_black_frame(
                dst_file,
                reference_size,
                camera_name,
                date_str,
                time_str,
            )

            created += 1

            print(
                f"Created black frame -> "
                f"{dst_file.name}"
            )

        current += timedelta(hours=2)

    print("\nDone.")
    print(f"Copied images : {copied}")
    print(f"Created blanks: {created}")
    print(f"Output folder : {output_dir}")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Flatten a camera/date/time folder structure. "
            "Optionally create a complete 2-hourly "
            "timeseries with black placeholder images."
        )
    )

    parser.add_argument(
        "input_directory",
        type=Path,
        help="Camera directory containing date folders."
    )

    parser.add_argument(
        "--full_timeseries",
        nargs=2,
        metavar=("START_DATE", "END_DATE"),
        help=(
            "Create a complete 2-hourly timeseries "
            "between START_DATE and END_DATE. "
            "Format: YYYY-MM-DD YYYY-MM-DD"
        )
    )

    args = parser.parse_args()

    collect_frames(
        args.input_directory,
        args.full_timeseries,
    )


if __name__ == "__main__":
    main()