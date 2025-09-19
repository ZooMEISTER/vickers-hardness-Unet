# -*- coding: utf-8 -*-
"""
plot_history.py — visualize training curves from history.json
Features:
- Plot train/val loss, val Dice, val IoU, learning rate
- Optional moving average smoothing
- Save as PNG in the same directory
- Optional export as CSV

Usage:
    python plot_history.py runs/unet_r34_384/history.json
    # or just the folder containing history.json
    python plot_history.py runs/unet_r34_384 --smooth 3 --csv
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

def moving_average(xs, k):
    """Simple moving average (window size = k)"""
    if k is None or k <= 1 or k > len(xs):
        return xs
    out, window, s = [], [], 0.0
    for v in xs:
        window.append(v)
        s += v
        if len(window) > k:
            s -= window.pop(0)
        out.append(s / len(window))
    return out

def load_history(path: Path):
    """Load history.json (or find it inside the given folder)"""
    p = Path(path)
    if p.is_dir():
        p = p / "history.json"
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        hist = json.load(f)
    if not isinstance(hist, list) or len(hist) == 0:
        raise ValueError("history.json must be a list of dicts")
    return p, hist

def to_series(hist):
    """Convert list of dicts into named lists"""
    def col(name): return [rec.get(name, None) for rec in hist]
    keys = ["epoch","train_loss","val_loss","val_dice","val_iou","lr","time"]
    series = {k: col(k) for k in keys}
    # Fill missing values
    for k in keys:
        last = 0.0
        filled = []
        for v in series[k]:
            if v is None: v = last
            filled.append(v)
            last = v
        series[k] = filled
    return series

def save_csv(series, out_csv: Path):
    import csv
    keys = ["epoch","train_loss","val_loss","val_dice","val_iou","lr","time"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(series["epoch"])):
            w.writerow([series[k][i] for k in keys])

def plot_curves(series, smooth=1, out_png: Path = None, show=False, title=None):
    ep = series["epoch"]
    S = lambda xs: moving_average(xs, smooth)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=120)
    fig.suptitle(title or "Training Curves", fontsize=14)

    ax = axes[0,0]
    ax.plot(ep, S(series["train_loss"]), label="train loss")
    ax.plot(ep, S(series["val_loss"]),   label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.grid(True, ls="--", alpha=0.4); ax.legend()

    ax = axes[0,1]
    ax.plot(ep, S(series["val_dice"]), label="val dice")
    ax.set_xlabel("epoch"); ax.set_ylabel("dice")
    ax.set_ylim(0, 1)
    ax.grid(True, ls="--", alpha=0.4); ax.legend()

    ax = axes[1,0]
    ax.plot(ep, S(series["val_iou"]), label="val iou")
    ax.set_xlabel("epoch"); ax.set_ylabel("iou")
    ax.set_ylim(0, 1)
    ax.grid(True, ls="--", alpha=0.4); ax.legend()

    ax = axes[1,1]
    ax.plot(ep, S(series["lr"]), label="learning rate")
    ax.set_xlabel("epoch"); ax.set_ylabel("lr")
    ax.grid(True, ls="--", alpha=0.4); ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight")
        print(f"[OK] Saved figure -> {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Visualize training curves from history.json")
    ap.add_argument("path", type=str,
                    help="Path to history.json or its folder")
    ap.add_argument("--smooth", type=int, default=1,
                    help="Moving average window size (>=2 for smoothing)")
    ap.add_argument("--show", action="store_true",
                    help="Show the plot window")
    ap.add_argument("--csv", action="store_true",
                    help="Also export history.csv")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PNG path (default: curves.png in the same folder)")
    args = ap.parse_args()

    hist_path, hist = load_history(Path(args.path))
    series = to_series(hist)

    out_dir = hist_path.parent
    out_png = Path(args.out) if args.out else (out_dir / "curves.png")

    if args.csv:
        save_csv(series, out_dir / "history.csv")
        print(f"[OK] Saved CSV   -> {out_dir/'history.csv'}")

    plot_curves(series, smooth=max(1, args.smooth),
                out_png=out_png, show=args.show,
                title=f"Training Curves — {out_dir.name}")

if __name__ == "__main__":
    main()
