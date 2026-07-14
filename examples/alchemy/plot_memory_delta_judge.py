#!/usr/bin/env python3
"""Plot offline DeepSeek memory-delta judge results without matplotlib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RUN_ORDER = ["act119", "co99", "sig4norm119_rep1"]
RUN_LABEL = {
    "act119": "ACT-only",
    "co99": "Co-train",
    "sig4norm119_rep1": "Sig4 norm-improve",
}
RUN_COLOR = {
    "act119": "#2f6fbb",
    "co99": "#d9792b",
    "sig4norm119_rep1": "#3b8f5a",
}


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


F10 = font(10)
F11 = font(11)
F12 = font(12)
F14 = font(14)
F16B = font(16, bold=True)
F18B = font(18, bold=True)


def rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def blend(a, b, t: float):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def score_color(v: float):
    v = max(0.0, min(1.0, float(v)))
    stops = [(68, 1, 84), (49, 104, 142), (53, 183, 121), (253, 231, 37)]
    pos = v * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    return blend(stops[i], stops[i + 1], pos - i)


def diff_color(v: float):
    v = max(-1.0, min(1.0, float(v)))
    if v < 0:
        return blend((49, 91, 172), (245, 245, 245), v + 1.0)
    return blend((245, 245, 245), (180, 35, 35), v)


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("judge") and "explore_score" in row["judge"]:
            rows.append(row)
    return rows


def score(row: dict) -> float:
    return float(row["judge"]["explore_score"])


def make_tables(rows: list[dict]):
    scores = {}
    for row in rows:
        scores[(row["run"], int(row["episode_file_index"]), int(row["trial_index"]))] = score(row)
    episodes = sorted({int(r["episode_file_index"]) for r in rows})
    trials = sorted({int(r["trial_index"]) for r in rows})
    runs = [r for r in RUN_ORDER if any(row["run"] == r for row in rows)]
    return runs, episodes, trials, scores


def line_xy(left, top, width, height, trials, t, y):
    x = left + (t - min(trials)) / (max(trials) - min(trials)) * width
    yy = top + (1.0 - y) * height
    return int(round(x)), int(round(yy))


def draw_axes(draw, left, top, width, height, trials, ylabel=False):
    axis = (55, 55, 55)
    grid = (220, 220, 220)
    draw.line((left, top, left, top + height, left + width, top + height), fill=axis, width=1)
    for y in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = top + (1.0 - y) * height
        draw.line((left, yy, left + width, yy), fill=grid, width=1)
        if ylabel:
            draw.text((left - 38, yy - 7), f"{y:.2f}", fill=(70, 70, 70), font=F10)
    for t in trials:
        x, _ = line_xy(left, top, width, height, trials, t, 0)
        draw.line((x, top + height, x, top + height + 4), fill=axis, width=1)
        draw.text((x - 4, top + height + 8), str(t), fill=(50, 50, 50), font=F10)


def plot_trend_lines(rows: list[dict], out: Path):
    runs, episodes, trials, scores = make_tables(rows)
    W, H = 1600, 520
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((W // 2 - 270, 18), "Within-episode trend: memory-delta reward over trials", fill=(25, 25, 25), font=F18B)
    panel_w = 450
    gap = 55
    left0 = 70
    top = 90
    plot_h = 320
    plot_w = 360
    for idx, run in enumerate(runs):
        left = left0 + idx * (panel_w + gap)
        color = rgb(RUN_COLOR[run])
        draw.text((left + 105, 58), RUN_LABEL[run], fill=(25, 25, 25), font=F16B)
        draw.rectangle((left, top, left + plot_w * 3 / 7, top + plot_h), fill=(232, 238, 247, 120))
        draw.rectangle((left + plot_w * 4 / 7, top, left + plot_w, top + plot_h), fill=(248, 238, 226, 120))
        draw.text((left + 75, top + 8), "early", fill=(70, 98, 127), font=F11)
        draw.text((left + 260, top + 8), "late", fill=(138, 91, 40), font=F11)
        draw_axes(draw, left, top, plot_w, plot_h, trials, ylabel=(idx == 0))
        mat = []
        for ep in episodes:
            ys = [scores[(run, ep, t)] for t in trials]
            mat.append(ys)
            pts = [line_xy(left, top, plot_w, plot_h, trials, t, y) for t, y in zip(trials, ys)]
            draw.line(pts, fill=color + (85,), width=2)
        mean = np.mean(np.array(mat), axis=0)
        pts = [line_xy(left, top, plot_w, plot_h, trials, t, y) for t, y in zip(trials, mean)]
        draw.line(pts, fill=color + (255,), width=4)
        for x, y in pts:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color + (255,))
        draw.text((left + 138, top + plot_h + 34), "Trial index", fill=(50, 50, 50), font=F12)
    draw.text((12, top + 105), "Judge score", fill=(50, 50, 50), font=F12)
    img.save(out)


def plot_early_late(rows: list[dict], out: Path):
    runs, episodes, trials, scores = make_tables(rows)
    stats = []
    for run in runs:
        ep_early, ep_late = [], []
        for ep in episodes:
            early = [scores[(run, ep, t)] for t in trials if t <= 4]
            late = [scores[(run, ep, t)] for t in trials if t >= 5]
            ep_early.append(float(np.mean(early)))
            ep_late.append(float(np.mean(late)))
        stats.append((run, float(np.mean(ep_early)), float(np.mean(ep_late)), float(np.mean(np.array(ep_late) - np.array(ep_early)))))

    W, H = 920, 500
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((230, 20), "Early vs late memory-delta reward inside each episode", fill=(25, 25, 25), font=F18B)
    left, top, plot_w, plot_h = 95, 80, 740, 310
    draw_axes(draw, left, top, plot_w, plot_h, [1, 2, 3], ylabel=True)
    max_y = 0.75
    bar_w = 58
    centers = np.linspace(left + 120, left + plot_w - 120, len(stats))
    for cx, (run, early, late, delta) in zip(centers, stats):
        for offset, val, color, label in [(-bar_w / 2, early, (109, 143, 199), "early"), (bar_w / 2, late, (216, 155, 92), "late")]:
            h = val / max_y * plot_h
            x0 = int(cx + offset - bar_w / 2)
            x1 = int(cx + offset + bar_w / 2)
            y0 = int(top + plot_h - h)
            draw.rectangle((x0, y0, x1, top + plot_h), fill=color)
            draw.text((x0 + 10, y0 - 17), f"{val:.3f}", fill=(40, 40, 40), font=F10)
        draw.text((int(cx) - 74, top + plot_h + 35), RUN_LABEL[run], fill=(40, 40, 40), font=F12)
        draw.text((int(cx) - 54, top + plot_h + 55), f"late-early {delta:+.3f}", fill=(60, 60, 60), font=F11)
    draw.rectangle((650, 87, 670, 103), fill=(109, 143, 199))
    draw.text((678, 84), "early trials 1-4", fill=(45, 45, 45), font=F11)
    draw.rectangle((650, 112, 670, 128), fill=(216, 155, 92))
    draw.text((678, 109), "late trials 5-8", fill=(45, 45, 45), font=F11)
    img.save(out)


def draw_heatmap_base(title, rows, out, mode):
    runs, episodes, trials, scores = make_tables(rows)
    W, H = 1550, 520
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((W // 2 - 240, 18), title, fill=(25, 25, 25), font=F18B)
    cell_w, cell_h = 45, 42
    top = 100
    left0 = 95
    panel_w = cell_w * len(trials) + 90
    gap = 55

    panels = runs if mode == "score" else [("act119", "co99"), ("act119", "sig4norm119_rep1"), ("co99", "sig4norm119_rep1")]
    for pidx, panel in enumerate(panels):
        left = left0 + pidx * (panel_w + gap)
        if mode == "score":
            run = panel
            ptitle = RUN_LABEL[run]
        else:
            a, b = panel
            if a not in runs or b not in runs:
                continue
            ptitle = f"{RUN_LABEL[a]} - {RUN_LABEL[b]}"
        draw.text((left + 35, 65), ptitle, fill=(25, 25, 25), font=F14)
        for j, t in enumerate(trials):
            draw.text((left + 53 + j * cell_w, top - 22), str(t), fill=(50, 50, 50), font=F10)
        for i, ep in enumerate(episodes):
            draw.text((left - 42, top + i * cell_h + 14), f"ep{ep}", fill=(50, 50, 50), font=F10)
            for j, t in enumerate(trials):
                if mode == "score":
                    val = scores[(run, ep, t)]
                    c = score_color(val)
                    txt = f"{val:.2f}"
                else:
                    val = scores[(a, ep, t)] - scores[(b, ep, t)]
                    c = diff_color(val)
                    txt = f"{val:+.2f}"
                x0 = left + j * cell_w
                y0 = top + i * cell_h
                draw.rectangle((x0, y0, x0 + cell_w - 2, y0 + cell_h - 2), fill=c)
                text_col = (255, 255, 255) if (mode == "score" and val < 0.55) else (35, 35, 35)
                draw.text((x0 + 8, y0 + 14), txt, fill=text_col, font=F10)
        draw.text((left + 145, top + len(episodes) * cell_h + 28), "Trial", fill=(50, 50, 50), font=F12)
    img.save(out)


def plot_winner_map(rows: list[dict], out: Path):
    runs, episodes, trials, scores = make_tables(rows)
    colors = {
        "tie": (217, 217, 217),
        "act119": rgb(RUN_COLOR["act119"]),
        "co99": rgb(RUN_COLOR["co99"]),
        "sig4norm119_rep1": rgb(RUN_COLOR["sig4norm119_rep1"]),
    }
    short = {"tie": "T", "act119": "ACT", "co99": "CO", "sig4norm119_rep1": "S4"}
    W, H = 760, 480
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((210, 22), "Winner per matched (episode, trial)", fill=(25, 25, 25), font=F18B)
    left, top = 110, 90
    cell_w, cell_h = 60, 45
    for j, t in enumerate(trials):
        draw.text((left + j * cell_w + 24, top - 25), str(t), fill=(50, 50, 50), font=F11)
    for i, ep in enumerate(episodes):
        draw.text((left - 50, top + i * cell_h + 14), f"ep{ep}", fill=(50, 50, 50), font=F11)
        for j, t in enumerate(trials):
            vals = {r: scores[(r, ep, t)] for r in runs}
            maxv = max(vals.values())
            winners = [r for r, v in vals.items() if abs(v - maxv) < 1e-9]
            winner = winners[0] if len(winners) == 1 else "tie"
            x0 = left + j * cell_w
            y0 = top + i * cell_h
            draw.rectangle((x0, y0, x0 + cell_w - 2, y0 + cell_h - 2), fill=colors[winner])
            draw.text((x0 + 17, y0 + 15), short[winner], fill=(255, 255, 255) if winner != "tie" else (20, 20, 20), font=F11)
    legend = [("ACT", colors["act119"]), ("CO", colors["co99"]), ("S4", colors["sig4norm119_rep1"]), ("Tie", colors["tie"])]
    lx, ly = 110, 345
    for i, (name, c) in enumerate(legend):
        draw.rectangle((lx + i * 135, ly, lx + i * 135 + 22, ly + 16), fill=c)
        draw.text((lx + i * 135 + 30, ly - 1), name, fill=(45, 45, 45), font=F12)
    draw.text((left + 205, top + len(episodes) * cell_h + 24), "Trial", fill=(50, 50, 50), font=F12)
    img.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs.jsonl from validate_memory_delta_judge.py")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    out_dir = Path(args.output_dir) if args.output_dir else pairs_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(pairs_path)
    if not rows:
        raise SystemExit(f"No judged rows found in {pairs_path}")

    plot_trend_lines(rows, out_dir / "within_episode_trend_lines.png")
    plot_early_late(rows, out_dir / "early_vs_late_bars.png")
    draw_heatmap_base("Matched episode-trial scores by run", rows, out_dir / "matched_score_heatmaps.png", mode="score")
    draw_heatmap_base("Matched comparison: pairwise score differences", rows, out_dir / "matched_pairwise_differences.png", mode="diff")
    plot_winner_map(rows, out_dir / "matched_winner_map.png")
    print(f"Wrote figures to {out_dir}")
    for p in sorted(out_dir.glob("*.png")):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
