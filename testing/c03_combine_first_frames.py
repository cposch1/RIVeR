#!/usr/bin/env python3

from pathlib import Path
from PIL import Image
import argparse


def extract_timestamp(path: Path) -> str:
    """
    Extract YYYYMMDD_HHMMSS from filenames like:

    ilh-cam1_20250723_040000.jpg
    ilh-cam2_20250723_040000.jpg
    """
    parts = path.stem.split("_")
    return "_".join(parts[-2:])


def combine_folders(
    folder1: Path,
    folder2: Path,
) -> None:

    # Derive site name automatically:
    # ilh-cam1*  -> ilh
    # lev5-cam2* -> lev5
    prefix = folder1.name.split("-cam")[0]

    out_name = f"{prefix}-cams"

    out_dir = folder1.parent / f"{out_name}_comb"
    out_dir.mkdir(exist_ok=True)

    files1 = {
        extract_timestamp(f): f
        for f in folder1.glob("*.jpg")
    }

    files2 = {
        extract_timestamp(f): f
        for f in folder2.glob("*.jpg")
    }

    timestamps = sorted(
        set(files1.keys()) & set(files2.keys())
    )

    print(f"Found {len(timestamps)} matching timestamps")

    for ts in timestamps:

        img1 = Image.open(files1[ts]).convert("RGB")
        img2 = Image.open(files2[ts]).convert("RGB")

        # Match heights
        height = max(img1.height, img2.height)

        if img1.height != height:
            new_w = int(img1.width * height / img1.height)
            img1 = img1.resize((new_w, height))

        if img2.height != height:
            new_w = int(img2.width * height / img2.height)
            img2 = img2.resize((new_w, height))

        combined = Image.new(
            "RGB",
            (img1.width + img2.width, height),
            "black",
        )

        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width, 0))

        outfile = out_dir / f"{out_name}_{ts}.jpg"

        combined.save(
            outfile,
            quality=95,
        )

        print(f"Saved {outfile.name}")

    print("\nDone")
    print(f"Output: {out_dir}")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Combine two flattened camera image folders "
            "side-by-side using matching timestamps."
        )
    )

    parser.add_argument(
        "folder1",
        type=Path,
        help="First flattened camera folder",
    )

    parser.add_argument(
        "folder2",
        type=Path,
        help="Second flattened camera folder",
    )

    args = parser.parse_args()

    combine_folders(
        args.folder1,
        args.folder2,
    )


if __name__ == "__main__":
    main()