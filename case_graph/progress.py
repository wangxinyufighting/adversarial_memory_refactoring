import sys
from dataclasses import dataclass, field
from typing import TextIO


@dataclass
class ProgressBar:
    """Minimal terminal progress bar for long-running extraction jobs."""

    label: str
    width: int = 30
    stream: TextIO = field(default_factory=lambda: sys.stderr)

    def update(self, current: int, total: int, suffix: str = "") -> None:
        total = max(total, 1)
        current = min(max(current, 0), total)
        ratio = current / total
        filled = int(round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        percent = int(round(ratio * 100))
        suffix_text = f" {suffix}" if suffix else ""
        end = "\n" if current >= total else ""
        self.stream.write(
            f"\r{self.label} [{bar}] {current}/{total} {percent}%{suffix_text}"
            + end
        )
        self.stream.flush()
