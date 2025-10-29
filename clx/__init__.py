from dotenv import load_dotenv

from .utils import (
    init_django,
)

load_dotenv()

__all__ = ["init_django"]
