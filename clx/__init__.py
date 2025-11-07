# flake8: noqa: E402
import json
from pathlib import Path

from dotenv import load_dotenv

config_path = Path.home() / ".cache" / "clx" / "config.json"
if config_path.exists():
    config = json.loads(config_path.read_text())
    if config.get("autoload-env"):
        load_dotenv(override=False)

from .utils import (
    download_file,
    extract_attachments,
    extract_bz2_file,
    extract_from_pattern,
    generate_hash,
    init_django,
    pd_save_or_append,
)

__all__ = [
    "download_file",
    "extract_bz2_file",
    "init_django",
    "pd_save_or_append",
    "extract_from_pattern",
    "extract_attachments",
    "generate_hash",
    "autoload_env",
]
