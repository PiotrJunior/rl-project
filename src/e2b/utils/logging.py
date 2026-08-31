"""Per-run logging to JSONL + CSV.

One run writes:

  <run_dir>/config.yaml    the fully resolved config (for reproduction)
  <run_dir>/eval.csv       one row per evaluation point (the learning curve)
  <run_dir>/diagnostics.csv one row per diagnostic point (exploration internals)
  <run_dir>/episodes.jsonl  one line per finished training episode
  <run_dir>/result.json    final summary, written once at the end

Keeping the learning curve and the exploration diagnostics in separate files
means the aggregation script can load just the curves (small) without touching
the much larger diagnostics when producing the headline plots.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


class CsvLogger:
    """Append-only CSV writer that discovers its header from the first row.

    Rows after the first are projected onto the header, so a caller adding a
    new key mid-run cannot silently corrupt the file.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None
        self._fh = None
        self._writer: csv.DictWriter | None = None
        if self.path.exists():
            self.path.unlink()

    def write(self, row: Mapping[str, Any]) -> None:
        if self._writer is None:
            self._fields = list(row.keys())
            self._fh = self.path.open("w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._writer.writeheader()
        assert self._fields is not None
        projected = {k: row.get(k, "") for k in self._fields}
        self._writer.writerow(projected)
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self._fh = self.path.open("w")

    def write(self, record: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class RunLogger:
    """Bundle of the loggers a single training run needs."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.eval = CsvLogger(self.run_dir / "eval.csv")
        self.diagnostics = CsvLogger(self.run_dir / "diagnostics.csv")
        self.episodes = JsonlLogger(self.run_dir / "episodes.jsonl")

    def write_result(self, result: Mapping[str, Any]) -> None:
        (self.run_dir / "result.json").write_text(json.dumps(result, indent=2))

    def close(self) -> None:
        self.eval.close()
        self.diagnostics.close()
        self.episodes.close()
