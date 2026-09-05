"""Progress reporting for the long stages.

Every stage runs for minutes, often unattended with stdout redirected to a log.
The bar therefore holds a single line and repaints it in place with a carriage
return whether or not stdout is a terminal -- a log stays one live line under
`tail -f` instead of scrolling hundreds of snapshots. Each stage leaves exactly
one finished line behind, so a whole run reads as a handful of lines.
"""

from __future__ import annotations

import shutil
import sys
import time
from typing import Optional, TextIO


def format_hms(seconds: float) -> str:
    """Compact duration: 42s, 7:13, 1:04:59."""
    if seconds != seconds or seconds < 0:            # NaN or negative
        return "--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    if m:
        return f"{m}:{s:02d}"
    return f"{s}s"


def human_hms(seconds: float) -> str:
    """Spoken-language duration: "45 sec", "2 mins and 27 sec", "1 hour and 4 mins"."""
    if seconds != seconds or seconds < 0:            # NaN or negative
        return "unknown"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    if h:
        head = f"{h} hour{'s' if h > 1 else ''}"
        return f"{head} and {m} min{'s' if m > 1 else ''}" if m else head
    if m:
        head = f"{m} min{'s' if m > 1 else ''}"
        return f"{head} and {sec} sec" if sec else head
    return f"{sec} sec"


class Progress:
    """A single-line bar with percentage, elapsed, ETA and projected total."""

    WIDTH = 28
    # Eighth-block glyphs give the bar a smooth leading edge as it grows.
    PARTIALS = " ▏▎▍▌▋▊▉"
    FULL, EMPTY = "█", "░"
    ASCII_FULL, ASCII_EMPTY = "#", "-"

    # 256-colour SGR. The fill ramps the way a phone charge indicator does --
    # red when barely started, green when nearly done -- painted on a white
    # channel so the hue reads at a glance against any terminal background.
    RESET = "\033[0m"
    WHITE_BG = "\033[48;5;231m"
    LABEL = "\033[1m"
    DIM = "\033[38;5;245m"
    DEVICE = {"CPU": "\033[38;5;39m", "GPU": "\033[38;5;213m"}
    RAMP = (
        "\033[38;5;196m",   # red
        "\033[38;5;208m",   # orange
        "\033[38;5;226m",   # yellow
        "\033[38;5;154m",   # light green
        "\033[38;5;46m",    # green
    )

    def __init__(
        self,
        total: int,
        label: str,
        stream: Optional[TextIO] = None,
        min_interval: float = 0.25,
        log_interval: float = 2.0,
        color: Optional[bool] = None,
        quiet: bool = False,
        device: str = "",
        unit: str = "item",
    ) -> None:
        self.total = max(0, int(total))
        self.label = label
        self.stream = stream if stream is not None else sys.stdout
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.interval = min_interval if self.tty else log_interval
        self.color = self.tty if color is None else bool(color)
        self.quiet = bool(quiet)
        self.device = str(device)
        self.unit = str(unit)
        self._last_width = 0
        self._bar_w = self.WIDTH
        self._rows = 0                  # lines currently held by the live block
        self.n = 0
        self.note = ""
        self._last_cells = -1
        self.start = time.perf_counter()
        self._last_paint = -1e9
        self._last_len = 0
        self._live = False
        self._unicode = self._probe_unicode()

    def _probe_unicode(self) -> bool:
        enc = getattr(self.stream, "encoding", None) or ""
        try:
            self.FULL.encode(enc or "utf-8")
            return True
        except (LookupError, UnicodeEncodeError):
            return False

    # -- lifecycle --------------------------------------------------------

    def __enter__(self) -> "Progress":
        self._paint(force=True)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def update(self, n: int = 1, note: str = "") -> None:
        self.n += n
        if note:
            self.note = note
        self._paint()

    def write(self, message: str) -> None:
        """Emit a standalone line without leaving a half-drawn bar behind."""
        self._clear()
        self.stream.write(message + "\n")
        self.stream.flush()
        self._paint(force=True)

    def close(self, summary: str = "") -> None:
        """Replace the live bar with one finished line.

        In quiet mode the bar simply vanishes, so a run's only output is the
        artefact it produced.
        """
        # Totals like max_nfev are upper bounds the work may finish under. The
        # count is what actually happened, so adopt it rather than leave the bar
        # frozen part-full, which reads as an abort.
        if 0 < self.n < self.total:
            self.total = self.n
            self._paint(force=True)
        self._clear()
        if self.quiet:
            self.stream.flush()
            return
        tail = f"  {summary}" if summary else ""
        body = (f"done {self.n}/{self.total} in {format_hms(self.elapsed)}{tail}")
        if self.color:
            dev = (f"{self.DEVICE.get(self.device, '')}{self.device:<3s}"
                   f"{self.RESET} " if self.device else "")
            self.stream.write(
                f"{self.LABEL}{self.label:<9s}{self.RESET} {dev}"
                f"{self.RAMP[-1]}✓{self.RESET} {self.DIM}{body}{self.RESET}\n"
            )
        else:
            tag = f"{self.device:<3s} " if self.device else ""
            self.stream.write(f"{self.label:<9s} {tag}{body}\n")
        self.stream.flush()

    # -- rendering --------------------------------------------------------

    def _width(self) -> int:
        """Current terminal width, re-read on every paint.

        A line longer than the pane wraps, and once it has wrapped a carriage
        return lands at the start of the LAST wrapped row rather than the line,
        so each repaint strands another row. Switching or resizing a pane
        changes the width underneath us, so it cannot be measured once.
        """
        if not self.tty:
            return 0                        # a file has no width to wrap at
        try:
            return max(40, shutil.get_terminal_size().columns)
        except OSError:
            return 100

    def _clear(self) -> None:
        """Blank the live block so the next write starts clean."""
        if not self._live:
            return
        if self.tty:
            # Cursor sits at the end of the last row; walk back up the block,
            # erasing each row, so a repaint never leaves a stale line behind.
            for _ in range(max(0, self._rows - 1)):
                self.stream.write("\r\033[2K\033[1A")
            self.stream.write("\r\033[2K")
        else:
            self.stream.write("\r" + " " * self._last_len + "\r")
        self._live = False
        self._rows = 0
        self._last_len = 0

    def _bar_fill(self, frac: float) -> str:
        w = self._bar_w
        if not self._unicode:
            return self.ASCII_FULL * int(round(w * frac))
        exact = w * frac
        full = int(exact)
        out = self.FULL * full
        if full < w:
            out += self.PARTIALS[int((exact - full) * len(self.PARTIALS))]
        return out[:w]

    def _bar(self, frac: float) -> str:
        fill = self._bar_fill(frac)
        empty = self.EMPTY if self._unicode else self.ASCII_EMPTY
        return (fill + empty * (self._bar_w - len(fill)))[:self._bar_w]

    def _detail_fields(self, elapsed: float, eta: float, projected: float,
                       rate: float) -> list:
        """Prose stats, ordered so a narrow pane sheds the least useful first."""
        noun = self.unit + ("s" if self.total != 1 else "")
        fields = [
            f"current {self.unit} {self.n} from total {self.total}",
            f"total runtime {human_hms(projected)}",
            f"remaining {human_hms(eta)}",
            f"elapsed {human_hms(elapsed)}",
            f"{rate:.1f} {noun}/sec",
        ]
        if self.note:
            fields.append(self.note)
        return fields

    def _paint(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_paint) < self.interval:
            return

        frac = min(max(self.n / self.total if self.total else 0.0, 0.0), 1.0)
        elapsed = self.elapsed
        # Repaint only when something visible actually changed: an eighth-cell
        # of bar, or the whole-second clock. Redrawing on every percent makes
        # the line flicker without telling the reader anything new.
        cells = int(self.WIDTH * 8 * frac)
        tick = (cells, int(elapsed))
        if not force and tick == self._last_cells:
            return
        self._last_cells = tick
        self._last_paint = now

        projected = elapsed / frac if frac > 1e-9 else float("nan")
        eta = projected - elapsed if projected == projected else float("nan")
        rate = self.n / elapsed if elapsed > 0 else 0.0

        width = self._width()
        if width and width != self._last_width:
            self._clear()
            self._last_width = width
            force = True

        head = 9 + 1 + (4 if self.device else 0)
        if width:
            self._bar_w = max(8, min(self.WIDTH, width - 1 - head - 8))
        else:
            self._bar_w = self.WIDTH

        fields = self._detail_fields(elapsed, eta, projected, rate)
        # A short indent, not one aligned under the bar: the prose line needs
        # the width more than it needs the alignment.
        indent = "  "
        sep = " \u00b7 "
        if width:
            budget = width - 1 - len(indent)
            while len(fields) > 1 and len(sep.join(fields)) > budget:
                fields.pop()
        detail = sep.join(fields)

        tag = f"{self.device:<3s} " if self.device else ""
        bar_plain = f"{self.label:<9s} {tag}{self._bar(frac)} {frac * 100:5.1f}%"

        if self.color:
            fill = self.RAMP[min(int(frac * len(self.RAMP)), len(self.RAMP) - 1)]
            dev = (f"{self.DEVICE.get(self.device, '')}{self.device:<3s}"
                   f"{self.RESET} " if self.device else "")
            bar_line = (f"{self.LABEL}{self.label:<9s}{self.RESET} {dev}"
                        f"{self.WHITE_BG}{fill}{self._bar_fill(frac)}"
                        f"{' ' * (self._bar_w - len(self._bar_fill(frac)))}"
                        f"{self.RESET} {frac * 100:5.1f}%")
            detail_line = f"{indent}{self.DIM}{detail}{self.RESET}"
        else:
            bar_line = bar_plain
            detail_line = f"{indent}{detail}"

        if self.tty:
            self._clear()
            self.stream.write(bar_line + "\n" + detail_line)
            self._rows = 2
            self._live = True
        else:
            # A log cannot address two rows, so collapse to one prose line.
            one = f"{bar_plain}  {detail}"
            pad = max(0, self._last_len - len(one))
            self.stream.write("\r" + one + " " * pad)
            self._last_len = len(one)
            self._live = True
            self._rows = 1
        self.stream.flush()


class StageTimer:
    """Times a named stage and prints one line when it ends."""

    def __init__(self, label: str, stream: Optional[TextIO] = None,
                 device: str = "") -> None:
        self.label = label
        self.device = device
        self.stream = stream if stream is not None else sys.stdout
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "StageTimer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self.start
        dev = f" on {self.device}" if self.device else ""
        self.stream.write(
            f"[{self.label}] total {format_hms(self.elapsed)}{dev}\n"
        )
        self.stream.flush()
