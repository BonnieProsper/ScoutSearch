# tests/test_cli.py

import unittest
import sys
from scout.cli import main

class TestCLI(unittest.TestCase):
    def test_cli_no_errors(self):
        """
        Ensure CLI runs without crashing.
        Provide dummy --records-file to avoid ValueError.
        """
        sys.argv = ["prog", "--query", "sample", "--records-file", "tests/data/dummy.json"]
        try:
            main()  # Should run without exceptions
        except SystemExit:
            # argparse calls sys.exit normally
            pass
        except Exception as e:
            self.fail(f"CLI raised an unexpected exception: {e}")


if __name__ == "__main__":
    unittest.main()
