#!/usr/bin/env python3
"""
monitor.py — Live ALSAT-EO-1 Training Monitor
==============================================
Reads training_live.json in real-time and shows a rich dashboard.

Usage (run in a SECOND terminal while training is running):
  python -m scripts.monitor
  python -m scripts.monitor --log results/training_autosave_42.json
  python -m scripts.monitor --variant full_system --seed 42
  python -m scripts.monitor --all        # watch all running variants
"""
from __future__ import annotations
import argparse, json, os, sys, time, glob
from pathlib import Path
from collections import deque
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    from rich.columns import Columns
    from rich import box
    from rich.progress import BarColumn, Progress, TextColumn
except ImportError:
    print("Install rich first:  pip install rich"); sys.exit(1)

ROOT      = Path(__file__).parent.parent
RESULTS   = ROOT / "results"
LOG_PATH  = RESULTS / "training_live.json"
console   = Console()
EVENT_ICONS = {"wildfire": "🔥", "flood": "🌊", "plume": "💨",
               "earthquake": "⚡", "eruption": "🌋"}


def load_log(path: Path) -> dict:
    try:
        with open(path) as f: return json.load(f)
    except: return {}


def bar(val: float, max_val: float, width: int = 20, color: str = "green") -> Text:
    filled = int(min(width, width * val / max(max_val, 1e-9)))
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * (width - filled), style="dim")
    return t


def make_dashboard(data: dict, path: Path) -> Panel:
    ep_rewards  = data.get("episode_rewards", [])
    dyn_success = data.get("ep_dyn_success", [])
    cf_rates    = data.get("ep_cf_rates", [])
    total_eps   = data.get("total_episodes", 2000)
    event_log   = data.get("event_log", [])      # list of per-episode dicts
    variant     = data.get("variant", "full_system")
    seed        = data.get("seed", 42)

    current_ep  = len(ep_rewards)
    last10_r    = ep_rewards[-10:] if ep_rewards else [0]
    avg10       = sum(last10_r) / max(len(last10_r), 1)
    last10_suc  = dyn_success[-10:] if dyn_success else [0]
    avg_suc     = sum(last10_suc) / max(len(last10_suc), 1)
    last_r      = ep_rewards[-1] if ep_rewards else 0

    grid = Table.grid(padding=(0, 2))
    grid.add_column(min_width=54); grid.add_column(min_width=40)

    # ── Left panel: progress + reward history ─────────────────────
    left = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    left.add_column("k", style="dim"); left.add_column("v")

    left.add_row("Variant",  Text(variant, style="bold yellow"))
    left.add_row("Seed",     str(seed))
    left.add_row("Episode",  f"{current_ep} / {total_eps}  ({100*current_ep//max(total_eps,1)}%)")
    left.add_row("Progress", bar(current_ep, total_eps, 30, "green"))
    left.add_row("", "")

    # Reward trend (last 10 eps as sparkline)
    left.add_row("Last reward", Text(f"{last_r:+.1f}", style="bold cyan"))
    left.add_row("avg10_r", Text(f"{avg10:+.1f}", style="green" if avg10 > 0 else "red"))
    left.add_row("Trend (10)", bar(max(avg10, 0), 250, 25,
                                    "green" if avg10 > 80 else "yellow" if avg10 > 30 else "red"))
    left.add_row("", "")
    left.add_row("Dyn success", Text(f"{avg_suc:.0%}",
                                        style="green" if avg_suc > .35 else "yellow"))
    left.add_row("Dyn bar",    bar(avg_suc, 1.0, 25, "cyan"))

    # ── Right panel: last events ──────────────────────────────────
    right = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    right.add_column("k", style="dim"); right.add_column("v")

    # Last 5 events from event_log
    if event_log:
        right.add_row(Text("LAST DYNAMIC EVENTS", style="bold yellow"), "")
        for ev in event_log[-5:][::-1]:
            icon  = EVENT_ICONS.get(ev.get("type", ""), "📍")
            imaged = ev.get("imaged", False)
            status = Text("IMAGED", style="bold green") if imaged else Text("MISSED", style="dim red")
            right.add_row(
                f"{icon} Ep{ev.get('ep','?')} {ev.get('type','?')}",
                Text(
                    f"lat={ev.get('lat',0):.1f} lon={ev.get('lon',0):.1f}  "
                    f"prio={ev.get('priority',0):.2f}  cloud={ev.get('cloud',0):.2f}  "
                    f"r={ev.get('reward',0):+.2f}",
                    style="cyan" if imaged else "dim"
                )
            )
            right.add_row("", status)
    else:
        right.add_row("Events", Text("(waiting for first episode...)", style="dim"))

    grid.add_row(Panel(left, title="[bold]Training Progress", border_style="yellow"),
                 Panel(right, title="[bold]Dynamic Events",   border_style="cyan"))

    ts = datetime.now().strftime("%H:%M:%S")
    mtime = ""
    try: mtime = f" | log updated {datetime.fromtimestamp(path.stat().st_mtime).strftime('%H:%M:%S')}"
    except: pass

    return Panel(
        grid,
        title=f"[bold yellow]ALSAT-EO-1 Phase 3 — Live Monitor[/]  {ts}{mtime}",
        border_style="yellow",
        subtitle=f"[dim]Watching: {path}  |  Press Ctrl+C to quit"
    )


def watch_single(path: Path, refresh: float = 3.0):
    console.print(f"\n[yellow]Watching:[/] {path}")
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            data = load_log(path)
            live.update(make_dashboard(data, path))
            time.sleep(refresh)


def watch_all(results_dir: Path, refresh: float = 5.0):
    """Watch all variant logs simultaneously in a summary table."""
    console.print("[yellow]Watching ALL variants...[/]")
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            tbl = Table(title="[bold yellow]ALSAT-EO-1 — All Ablation Variants",
                        box=box.ROUNDED, border_style="yellow")
            tbl.add_column("Variant", style="yellow")
            tbl.add_column("Seed")
            tbl.add_column("Episode")
            tbl.add_column("Last r")
            tbl.add_column("avg10_r")
            tbl.add_column("Dyn%")
            tbl.add_column("Status")

            for jf in sorted(results_dir.glob("*.json")):
                d = load_log(jf)
                if not d: continue
                eps  = d.get("episode_rewards", [])
                suc  = d.get("ep_dyn_success", [])
                n    = len(eps)
                last = eps[-1] if eps else 0
                avg  = sum(eps[-10:]) / max(len(eps[-10:]), 1) if eps else 0
                ds   = sum(suc[-10:]) / max(len(suc[-10:]), 1) if suc else 0
                total = d.get("total_episodes", 2000)
                done  = "[green]DONE" if n >= total else "[yellow]RUNNING"
                tbl.add_row(
                    d.get("variant", jf.stem), str(d.get("seed", "?")),
                    f"{n}/{total}",
                    Text(f"{last:+.0f}", style="cyan"),
                    Text(f"{avg:+.0f}", style="green" if avg > 80 else "yellow"),
                    Text(f"{ds:.0%}", style="green" if ds > .35 else "red"),
                    Text.from_markup(done)
                )
            live.update(tbl)
            time.sleep(refresh)


def main():
    ap = argparse.ArgumentParser(description="Live ALSAT-EO-1 Training Monitor")
    ap.add_argument("--log", default=str(LOG_PATH))
    ap.add_argument("--variant", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--refresh", type=float, default=3.0,
                    help="Refresh interval in seconds")
    args = ap.parse_args()

    if args.all:
        watch_all(RESULTS, args.refresh)
    else:
        log_path = Path(args.log)
        if args.variant:
            log_path = RESULTS / f"training_autosave_{args.seed}.json"
        if not log_path.exists():
            console.print(f"[red]Log not found:[/] {log_path}")
            console.print("Start training first, then run monitor in another terminal.")
            sys.exit(1)
        watch_single(log_path, args.refresh)


if __name__ == "__main__":
    main()