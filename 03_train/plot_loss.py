"""Draw the loss curve from a training log. No plotting library.

    python 03_train/plot_loss.py

Reads `runs/latest/log.jsonl` and writes an SVG. Matplotlib would be one import
and forty lines less, and it would also be the only heavyweight dependency in
this repository — for a line, some axes and a bit of text. SVG is text. We can
write text.

The output is deliberately plain and works on both GitHub themes: no
background fill, one accent colour that reads on light and dark, everything
else in a mid grey that neither theme swallows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

WIDTH, HEIGHT = 760, 360
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 64, 24, 36, 44
ACCENT = "#2f81f7"  # readable on white and on GitHub's dark background
GREY = "#8b949e"


def read_log(path: Path) -> list[tuple[int, float]]:
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "step" in record and "loss" in record:
            points.append((record["step"], record["loss"]))
    # A log is appended to across runs, so the same step can appear twice.
    # Keep the last value for each step and sort — otherwise the line folds
    # back on itself and the picture is a lie.
    latest = dict(points)
    return sorted(latest.items())


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick values that cover [low, high]."""
    if high <= low:
        return [low]
    raw = (high - low) / count
    magnitude = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 0.1
    step = max(magnitude, round(raw / magnitude) * magnitude)
    start = step * int(low / step)
    ticks = []
    value = start
    while value <= high + step / 2:
        if value >= low - step / 2:
            ticks.append(round(value, 4))
        value += step
    return ticks


def render(points: list[tuple[int, float]], title: str) -> str:
    steps = [s for s, _ in points]
    losses = [loss for _, loss in points]
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(losses), max(losses)
    span = max(y_max - y_min, 1e-6)
    y_min, y_max = y_min - span * 0.1, y_max + span * 0.1

    plot_w = WIDTH - PAD_LEFT - PAD_RIGHT
    plot_h = HEIGHT - PAD_TOP - PAD_BOTTOM

    def sx(step: int) -> float:
        return PAD_LEFT + (step - x_min) / max(x_max - x_min, 1) * plot_w

    def sy(loss: float) -> float:
        return PAD_TOP + (y_max - loss) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">',
        f'<text x="{PAD_LEFT}" y="22" font-size="13" fill="{GREY}">{title}</text>',
    ]

    for value in nice_ticks(y_min, y_max):
        y = sy(value)
        if not (PAD_TOP - 1 <= y <= PAD_TOP + plot_h + 1):
            continue
        parts.append(
            f'<line x1="{PAD_LEFT}" y1="{y:.1f}" x2="{PAD_LEFT + plot_w}" y2="{y:.1f}" '
            f'stroke="{GREY}" stroke-opacity="0.25" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_LEFT - 10}" y="{y + 4:.1f}" font-size="11" fill="{GREY}" '
            f'text-anchor="end">{value:.1f}</text>'
        )

    for step in (x_min, (x_min + x_max) // 2, x_max):
        parts.append(
            f'<text x="{sx(step):.1f}" y="{HEIGHT - PAD_BOTTOM + 20}" font-size="11" '
            f'fill="{GREY}" text-anchor="middle">{step}</text>'
        )

    parts.append(
        f'<text x="{PAD_LEFT + plot_w / 2:.0f}" y="{HEIGHT - 10}" font-size="11" '
        f'fill="{GREY}" text-anchor="middle">step</text>'
    )
    parts.append(
        f'<text x="16" y="{PAD_TOP + plot_h / 2:.0f}" font-size="11" fill="{GREY}" '
        f'text-anchor="middle" transform="rotate(-90 16 {PAD_TOP + plot_h / 2:.0f})">loss</text>'
    )

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{sx(step):.1f},{sy(loss):.1f}"
        for i, (step, loss) in enumerate(points)
    )
    parts.append(f'<path d="{path}" fill="none" stroke="{ACCENT}" stroke-width="2"/>')

    for step, loss in (points[0], points[-1]):
        parts.append(f'<circle cx="{sx(step):.1f}" cy="{sy(loss):.1f}" r="3.5" fill="{ACCENT}"/>')

    first, last = points[0], points[-1]
    parts.append(
        f'<text x="{sx(first[0]) + 8:.1f}" y="{sy(first[1]) - 8:.1f}" font-size="11" '
        f'fill="{ACCENT}">{first[1]:.2f}</text>'
    )
    parts.append(
        f'<text x="{sx(last[0]) - 8:.1f}" y="{sy(last[1]) + 18:.1f}" font-size="11" '
        f'fill="{ACCENT}" text-anchor="end">{last[1]:.2f}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=REPO_ROOT / "runs" / "latest" / "log.jsonl")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "assets" / "loss-curve.svg")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    if not args.log.exists():
        print(f"No log at {args.log}. Run chapter 03 first.")
        return 1

    points = read_log(args.log)
    if len(points) < 2:
        print(f"Only {len(points)} point(s) in the log — nothing to draw yet.")
        return 1

    title = args.title or (
        f"training loss, steps {points[0][0]}-{points[-1][0]} "
        f"({points[0][1]:.2f} to {points[-1][1]:.2f})"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(points, title), encoding="utf-8")
    print(f"Wrote {args.out} — {len(points)} points, {points[0][1]:.4f} to {points[-1][1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
