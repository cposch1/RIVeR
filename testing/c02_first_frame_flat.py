#!/usr/bin/env python3

from pathlib import Path
import os
import re
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

    font_size = 40
    font = None

    # DejaVu (Linux), Arial Bold (Windows); the default font as last resort
    for font_name in ("DejaVuSans-Bold.ttf", "arialbd.ttf"):
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except OSError:
            continue

    if font is None:
        try:
            font = ImageFont.load_default(size=font_size)  # Pillow >= 10.1
        except TypeError:
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


def to_path(path_str: str) -> Path:
    """
    Accept Windows paths (C:\\... or C:/...) and Git Bash paths (/c/...).
    """

    path_str = os.path.expanduser(path_str)

    # Git Bash style /c/Users/... -> C:/Users/... (Windows only)
    m = re.match(r"^/([a-zA-Z])(/|$)", path_str)
    if os.name == "nt" and m:
        path_str = f"{m.group(1).upper()}:/{path_str[3:]}"

    return Path(path_str)


def confirm(question: str) -> bool:
    try:
        return input(question).strip().lower() in ("y", "yes")
    except EOFError:  # no interactive input
        return False


DATE_RE = re.compile(r"^\d{8}$")
TIME_RE = re.compile(r"^\d{6}$")


def is_camera_dir(path: Path) -> bool:
    """A camera folder holds YYYYMMDD date folders."""
    return any(
        p.is_dir() and DATE_RE.match(p.name)
        for p in path.iterdir()
    )


def find_camera_dirs(root: Path) -> list:
    """All camera folders at any depth below (or equal to) root."""
    cameras = []
    for dirpath, dirnames, _ in os.walk(root):
        path = Path(dirpath)
        if is_camera_dir(path):
            cameras.append(path)
            dirnames[:] = []  # don't descend into date/time folders
        else:
            dirnames.sort()
    return sorted(cameras)


def old_outputs_for(output_dir: Path, camera_name: str) -> list:
    """Images a previous run wrote for this camera."""
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob(f"{camera_name}_*.jpg"))


def collect_frames(
    input_dir: Path,
    full_timeseries=None,
    overwrite: bool = False,
) -> None:
    """
    Input: a single camera folder, or any folder tree containing camera
    folders (e.g. site/camera/...):

    input/
    └── ilhh/
        └── ilh-cam1-pt/
            └── YYYYMMDD/
                └── HHMMSS/
                    └── *.jpg

    Output: next to the input, same tree with only the date and time
    levels collapsed:

    input_flat_frames/
    └── ilhh/
        └── ilh-cam1-pt/
            └── ilh-cam1-pt_YYYYMMDD_HHMMSS.jpg
    """

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    input_dir = input_dir.resolve()
    output_root = (
        input_dir.parent /
        f"{input_dir.name}_flat_frames"
    )

    cameras = find_camera_dirs(input_dir)
    if not cameras:
        raise RuntimeError(
            f"No camera folders (with YYYYMMDD subfolders) found in {input_dir}"
        )

    jobs = [
        (cam, output_root / cam.relative_to(input_dir))
        for cam in cameras
    ]

    print(f"Found {len(jobs)} camera folder(s):")
    for cam, _ in jobs:
        print(f"  {cam.relative_to(input_dir.parent)}")

    # Images from an earlier run are only replaced after confirmation
    n_old = sum(
        len(old_outputs_for(out, cam.name)) for cam, out in jobs
    )
    if n_old and not overwrite:
        if not confirm(
            f"{output_root} already contains {n_old} images.\n"
            f"Overwrite them completely? (y/n): "
        ):
            print("Aborted, nothing changed.")
            return

    for cam, out in jobs:
        print(f"\n=== {cam.relative_to(input_dir.parent)} ===")
        flatten_camera(cam, out, full_timeseries)

    print(f"\nAll done. Output folder: {output_root}")


def flatten_camera(
    input_dir: Path,
    output_dir: Path,
    full_timeseries=None,
) -> None:
    """Collapse the date/time levels of one camera folder into output_dir."""

    camera_name = input_dir.name
    old_outputs = old_outputs_for(output_dir, camera_name)

    existing_images = {}

    reference_size = None

    # -----------------------------------------
    # Scan source imagery
    # -----------------------------------------

    for date_dir in sorted(input_dir.iterdir()):

        if not date_dir.is_dir() or not DATE_RE.match(date_dir.name):
            continue

        date_str = date_dir.name

        for time_dir in sorted(date_dir.iterdir()):

            if not time_dir.is_dir() or not TIME_RE.match(time_dir.name):
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
        print(f"No images found in {input_dir}, skipped.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear the earlier run only now that there is something to replace it
    for old_file in old_outputs:
        old_file.unlink()

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
            "Collapse the date/time folder levels of one or more camera "
            "folders into <input>_flat_frames, keeping the rest of the "
            "folder structure. Optionally create a complete 2-hourly "
            "timeseries with black placeholder images."
        )
    )

    parser.add_argument(
        "input_directory",
        type=to_path,
        help=(
            "A camera folder (with YYYYMMDD subfolders) or any folder "
            "containing camera folders at any depth "
            "(Windows C:/... or Git Bash /c/... path)."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace images from an earlier run without asking."
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
        args.overwrite,
    )


if __name__ == "__main__":
    main()