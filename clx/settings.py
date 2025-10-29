import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = Path(os.getenv("LOCAL_DATA_DIR", BASE_DIR.parent / "data"))
