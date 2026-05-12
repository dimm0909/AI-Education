from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from src.data.download_pretrained_model import (
    DEFAULT_PUBLIC_KEY,
    build_metadata_url,
    extract_download_href,
)


def test_build_metadata_url_contains_encoded_public_key() -> None:
    url = build_metadata_url(DEFAULT_PUBLIC_KEY)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert "cloud-api.yandex.net" in parsed.netloc
    assert query.get("public_key") == [DEFAULT_PUBLIC_KEY]


def test_extract_download_href_success() -> None:
    href = extract_download_href({"href": "https://downloader.disk.yandex.ru/some-temporary-link"})
    assert href.startswith("https://")


def test_extract_download_href_raises_when_missing() -> None:
    with pytest.raises(RuntimeError):
        extract_download_href({"method": "GET"})
