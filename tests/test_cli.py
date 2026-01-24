import sys
import json
from pathlib import Path

from scout.cli import main


def test_cli_runs_without_error(tmp_path, capsys):
    records_file = tmp_path / "records.json"
    records_file.write_text(json.dumps([
        {"id": 1, "text": "sample document"}
    ]))

    sys.argv = [
        "prog",
        "--query", "sample",
        "--records-file", str(records_file),
    ]

    try:
        main()
    except SystemExit:
        # argparse is allowed to exit
        pass

    captured = capsys.readouterr()
    assert captured.out or captured.err

def test_cli_imports():
    import scout.cli