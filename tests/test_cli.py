# tests/test_cli.py
import unittest
import sys
from scout.cli import main

class TestCLI(unittest.TestCase):
    def test_cli_no_errors(self):
        sys.argv = ["prog", "--query", "sample"]
        try:
            main()  # Should run without exceptions
        except SystemExit:
            # argparse will call sys.exit() normally
            pass

if __name__ == "__main__":
    unittest.main()
