from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

YANDEX_DOWNLOAD_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
DEFAULT_PUBLIC_KEY = "https://disk.yandex.ru/d/fUbi6bAAbJJTOA"


def build_metadata_url(public_key: str) -> str:
    """Build Yandex Disk API URL that returns a one-time direct download link."""
    return f"{YANDEX_DOWNLOAD_API}?{urlencode({'public_key': public_key})}"


def extract_download_href(payload: dict[str, object]) -> str:
    """Extract direct download href from Yandex Disk API response payload."""
    href = payload.get("href")
    if not isinstance(href, str) or not href:
        raise RuntimeError(
            "Yandex Disk response does not contain a valid download URL in 'href'. "
            "Check that the public link is valid and download is allowed."
        )
    return href


def resolve_direct_download_url(public_key: str, timeout: int = 60) -> str:
    """Resolve a Yandex public link to a temporary direct file download URL."""
    metadata_url = build_metadata_url(public_key)
    request = Request(metadata_url, headers={"User-Agent": "BikeFlow/1.0"})

    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected Yandex Disk API response format")

    return extract_download_href(payload)


def download_pretrained_model(public_key: str, output_path: Path, timeout: int = 120) -> int:
    """Download pretrained model from Yandex Disk to the output path."""
    direct_url = resolve_direct_download_url(public_key=public_key, timeout=timeout)
    download_request = Request(direct_url, headers={"User-Agent": "BikeFlow/1.0"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(download_request, timeout=timeout) as response, output_path.open("wb") as f:
        shutil.copyfileobj(response, f)

    file_size = output_path.stat().st_size
    if file_size <= 0:
        raise RuntimeError(f"Downloaded file is empty: {output_path}")

    return file_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pretrained BikeFlow model from Yandex Disk")
    parser.add_argument("--public-key", type=str, default=DEFAULT_PUBLIC_KEY, help="Public Yandex Disk URL")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.joblib"), help="Output model path")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size = download_pretrained_model(args.public_key, args.output, timeout=args.timeout)
    print(f"Saved pretrained model to {args.output} ({size} bytes)")


if __name__ == "__main__":
    main()
