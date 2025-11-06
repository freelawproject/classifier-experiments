from pathlib import Path
from unittest import TestCase

from clx.settings import CLX_HOME, TESTING


class EnvTest(TestCase):
    """Ensure the testing environment is set up correctly."""

    def test_env(self):
        self.assertEqual(CLX_HOME, Path(__file__).parent / "fixtures" / "home")
        self.assertEqual(TESTING, True)
