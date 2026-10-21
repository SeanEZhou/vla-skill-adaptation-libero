"""M5: README demo GIFs. One GIF per policy, showing the same two episodes side by side:
the new task (LIBERO-90 task 56) and a known task (LIBERO-Object task 0), same init states for
every policy. Built from the per-episode rollout videos written by scripts/evaluate.py.

    uv run python scripts/make_demo_gifs.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vla_sim_poc.config import PROJECT_ROOT

FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf"
CONTROL_HZ = 20
INK, INK_2, SURFACE = (11, 11, 11), (82, 81, 78), (252, 252, 251)
GOOD, BAD = (12, 163, 12), (208, 59, 59)  # status good / critical, always paired with a symbol

TASKS = [
    ("New task: butter \u2192 tray", "libero_90_t56", "libero_90_t56_init00"),
    ("Known task: soup \u2192 basket", "libero_object_n10", "libero_object_t00_init00"),
]
POLICIES = [
    ("baseline", "Before fine-tuning (smolvla_libero)",
     {"libero_90_t56": "results/baseline/libero_90_t56", "libero_object_n10": "results/validation/libero_object_n10"}),
    ("expert_only", "Fine-tuned on the new task only (B1)",
     {k: f"results/finetuned/expert_only_s0/{k}" for k in ("libero_90_t56", "libero_object_n10")}),
    ("expert_only_rehearsal", "Fine-tuned with rehearsal (B2)",
     {k: f"results/finetuned/expert_only_rehearsal_s0/{k}" for k in ("libero_90_t56", "libero_object_n10")}),
]


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def load_episode(run_dir: Path, key: str):
    rec = next(json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if json.loads(l)["key"] == key)
    video = PROJECT_ROOT / rec["video_path"]
    frames = [f[:, : f.shape[1] // 2] for f in imageio.mimread(video, memtest=False)]  # agentview half
    return rec, frames


def panel(frame, label, status, color, size):
    img = Image.new("RGB", (size, size + 52), SURFACE)
    img.paste(Image.fromarray(frame).resize((size, size), Image.BILINEAR), (0, 22))
    d = ImageDraw.Draw(img)
    d.text((2, 3), label, fill=INK, font=font(13, bold=True))
    if color == INK_2:  # running clock
        d.text((2, size + 28), status, fill=color, font=font(14))
    else:  # outcome badge: filled so the colour survives GIF palette reduction; symbol + words carry meaning
        f = font(14, bold=True)
        w = d.textlength(status, font=f)
        d.rounded_rectangle((0, size + 25, w + 14, size + 49), radius=5, fill=color)
        d.text((7, size + 28), status, fill=(255, 255, 255), font=f)
    return img


def build(name, title, dirs, out: Path, speed: int, size: int, hold_s: float):
    eps = [load_episode(PROJECT_ROOT / dirs[k], key) for _, k, key in TASKS]
    n = max(len(f) for _, f in eps)
    hold = int(hold_s * CONTROL_HZ / speed)
    canvas_w = size * 2 + 24
    out_frames = []
    for t in list(range(0, n, speed)) + [n - 1] * hold:
        canvas = Image.new("RGB", (canvas_w, size + 52 + 34), SURFACE)
        ImageDraw.Draw(canvas).text((8, 7), title, fill=INK, font=font(16, bold=True))
        for j, ((label, _, _), (rec, frames)) in enumerate(zip(TASKS, eps)):
            i = min(t, len(frames) - 1)
            done = t >= len(frames) - 1
            secs = rec["steps"] / CONTROL_HZ
            if not done:
                status, color = f"t = {i / CONTROL_HZ:4.1f} s", INK_2
            elif rec["success"]:
                status, color = f"✓ success at {secs:.1f} s", GOOD
            else:
                status, color = f"✗ failed: timeout at {secs:.0f} s", BAD
            canvas.paste(panel(frames[i], label, status, color, size), (8 + j * (size + 8), 34))
        out_frames.append(np.asarray(canvas))

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "x.mp4"
        imageio.mimwrite(mp4, out_frames, fps=CONTROL_HZ, codec="libx264", quality=10, macro_block_size=1)
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-vf",
             f"fps={CONTROL_HZ},split[a][b];[a]palettegen=max_colors=160:stats_mode=full[p];"
             "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-loop", "0", str(out)],
            check=True,
        )
    outcomes = {k: rec["success"] for (_, k, _), (rec, _) in zip(TASKS, eps)}
    print(f"{name}: {out} ({out.stat().st_size / 1e6:.1f} MB, {len(out_frames)} frames) outcomes={outcomes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "docs/media"))
    ap.add_argument("--speed", type=int, default=2, help="playback speed-up (frame stride)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--hold-s", type=float, default=3.0, help="seconds to hold the final frame")
    args = ap.parse_args()
    for name, title, dirs in POLICIES:
        build(name, title, dirs, Path(args.out_dir) / f"demo_{name}.gif", args.speed, args.size, args.hold_s)


if __name__ == "__main__":
    main()
