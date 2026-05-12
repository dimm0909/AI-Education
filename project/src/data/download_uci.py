from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

UCI_BIKE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download UCI Bike Sharing dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--url", type=str, default=UCI_BIKE_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from: {args.url}")
    with urlopen(args.url) as response:
        archive_bytes = response.read()

    zip_path = args.output_dir / "bike_sharing.zip"
    zip_path.write_bytes(archive_bytes)

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        archive.extractall(args.output_dir)

    print(f"Downloaded and extracted to: {args.output_dir}")


if __name__ == "__main__":
    main()
