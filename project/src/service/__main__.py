from __future__ import annotations

import os

import uvicorn

from src.utils.config import load_yaml


def main() -> None:
    cfg = load_yaml(os.getenv("BIKEFLOW_SERVICE_CONFIG", "configs/service.yaml"))
    host = cfg.get("service", {}).get("host", "0.0.0.0")
    port = int(cfg.get("service", {}).get("port", 8000))

    uvicorn.run("src.service.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
