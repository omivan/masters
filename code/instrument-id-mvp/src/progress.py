"""
Overall progress of a long run, written to a small JSON file so it can be shown
outside the process (Claude Code status line, or `python src/progress.py` in a
second terminal for a live bar).

Units are arbitrary work items weighted by expected time (e.g. 1 per training
epoch, ~19 per test-set evaluation of one checkpoint); the ETA is extrapolated
from the measured rate, so it self-corrects after the first few units.
"""

import json
import sys
import time
from pathlib import Path

PROGRESS_FILE = Path.home() / ".claude" / "train_progress.json"


class Progress:
    def __init__(self, label: str, total_units: float):
        self.label = label
        self.total = float(total_units)
        self.done  = 0.0
        self.start = time.time()
        self.stage = ""
        self._write()

    def advance(self, units: float = 1.0, stage: str | None = None):
        self.done = min(self.total, self.done + units)
        if stage is not None:
            self.stage = stage
        self._write()

    def finish(self, stage: str = "done"):
        self.done, self.stage = self.total, stage
        self._write()

    def _write(self):
        elapsed = time.time() - self.start
        eta = elapsed / self.done * (self.total - self.done) if self.done else None
        state = dict(label=self.label, stage=self.stage, done=self.done, total=self.total,
                     pct=100.0 * self.done / self.total if self.total else 100.0,
                     elapsed_s=elapsed, eta_s=eta, updated=time.time())
        try:
            tmp = PROGRESS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(state), encoding="utf-8")
            tmp.replace(PROGRESS_FILE)
        except OSError:
            pass


def _fmt(sec) -> str:
    if sec is None:
        return "?"
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s"


def watch(interval: float = 2.0):
    """Live progress bar in the terminal until the run reaches 100%."""
    while True:
        try:
            st = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print("\rwaiting for progress file…", end="", flush=True)
            time.sleep(interval)
            continue
        width = 30
        fill = int(width * st["pct"] / 100)
        bar = "█" * fill + "░" * (width - fill)
        print(f"\r{st['label']}  {bar} {st['pct']:5.1f}%  "
              f"elapsed {_fmt(st['elapsed_s'])}  left ~{_fmt(st['eta_s'])}  | {st['stage'][:40]:<40}",
              end="", flush=True)
        if st["pct"] >= 100:
            print()
            return
        time.sleep(interval)


if __name__ == "__main__":
    watch(float(sys.argv[1]) if len(sys.argv) > 1 else 2.0)
