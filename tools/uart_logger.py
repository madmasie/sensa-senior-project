"""
uart_logger.py — Reads SEN55 data from serial UART, stores it in a
timestamped pandas DataFrame, and saves it as a .pkl file.

Recording can be controlled two ways:

  1. Keyboard commands typed into this program's terminal:
         start  (or  s)   — begin a recording session
         stop   (or  x)   — end the session and write the .pkl file
         quit   (or  q)   — flush any session and exit
         <Enter>          — toggle recording on/off
  2. Sentinel strings sent over UART by the firmware:
         START_RECORDING  — firmware signals that a session has begun
         STOP_RECORDING   — firmware signals that the session has ended

Each START/STOP pair produces one .pkl file. Multiple pairs in the same
run each produce their own file (the output filename is timestamped at
the moment the session is saved).

Dependencies: pyserial, pandas  (pip install pyserial pandas)

Usage:
    python uart_logger.py --port /dev/ttyUSB0 --baud 115200 --out ./data
    python uart_logger.py --port COM3 --out ./data

Expected serial line format (from firmware):
    [SEN55] PM1=1.23 PM2.5=2.34 PM4=3.45 PM10=4.56 Temp=25.1 RH=50.2 VOC=100.0 NOx=10.0
"""

import argparse
import queue
import re
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import serial


# ── regex that matches one SEN55 output line ──────────────────────────────────
# Captures every named float field regardless of whitespace variation.
_LINE_RE = re.compile(
    r"\[SEN55\]"
    r"\s+PM1=(?P<pm1>[\d.]+)"
    r"\s+PM2\.5=(?P<pm2_5>[\d.]+)"
    r"\s+PM4=(?P<pm4>[\d.]+)"
    r"\s+PM10=(?P<pm10>[\d.]+)"
    r"\s+Temp=(?P<temp>[\d.]+)"
    r"\s+RH=(?P<rh>[\d.]+)"
    r"\s+VOC=(?P<voc>[\d.]+)"
    r"\s+NOx=(?P<nox>[\d.]+)"
)


def parse_line(line: str) -> dict | None:
    """Return a dict of float values if *line* matches the SEN55 format, else None."""
    m = _LINE_RE.search(line)
    if m is None:
        return None
    return {k: float(v) for k, v in m.groupdict().items()}


def save_session(rows: list[dict], out_dir: Path) -> None:
    """Persist a completed recording session to a timestamped .pkl file."""
    if not rows:
        print("  [warn] Empty session — nothing written.")
        return

    # Timestamp the filename at the moment of saving
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sen55_{ts}.pkl"

    df = pd.DataFrame(rows).set_index("timestamp")

    # Explicit column order matching the firmware print order
    col_order = ["pm1", "pm2_5", "pm4", "pm10", "temp", "rh", "voc", "nox"]
    df = df[col_order]

    df.to_pickle(out_path)
    print(f"  Saved {len(df)} rows → {out_path}")


def _command_reader(cmd_queue: "queue.Queue[str]") -> None:
    """
    Background (daemon) thread: read whole lines from stdin and push each one,
    lower-cased and stripped, onto *cmd_queue* for the main loop to consume.

    Blocking on stdin here keeps the serial read loop responsive. The thread
    ends naturally on EOF (e.g. stdin closed or piped input exhausted).
    """
    for line in sys.stdin:
        cmd_queue.put(line.strip().lower())


def collect(port: str, baud: int, out_dir: Path) -> None:
    """
    Keep the serial port open indefinitely, recording SEN55 data into .pkl
    files. Recording starts/stops on either keyboard commands or the firmware
    sentinels START_RECORDING / STOP_RECORDING.

    State machine:
        WAITING  →  (start)  →  RECORDING
        RECORDING →  (stop)  →  WAITING   (file written here)

    Ctrl-C (or the `quit` command) flushes any in-progress session before exit.
    """
    recording = False   # True while a session is active
    rows: list[dict] = []
    stop = False        # Set to True by Ctrl-C or the `quit` command

    def _on_sigint(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    def begin(origin: str) -> None:
        """Start a new recording session; save any in-progress one first."""
        nonlocal recording, rows
        if recording:
            print("  [warn] Recording already active — saving current session first.")
            save_session(rows, out_dir)
        recording = True
        rows = []
        print(f"  [START] Recording began at {datetime.now():%H:%M:%S}  (via {origin})")

    def end(origin: str) -> None:
        """Stop the current session and write it to disk."""
        nonlocal recording, rows
        if not recording:
            print(f"  [warn] Stop ignored — not currently recording  (via {origin}).")
            return
        recording = False
        print(f"  [STOP]  Recording ended at {datetime.now():%H:%M:%S}  (via {origin})")
        save_session(rows, out_dir)
        rows = []
        print("\nIdle — type 'start' (or press Enter) to begin a new session.\n")

    # Background thread feeds keyboard commands to the main loop.
    cmd_queue: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_command_reader, args=(cmd_queue,), daemon=True).start()

    print(f"Listening on {port} at {baud} baud.")
    print("Commands:  start | s   stop | x   quit | q   (or press Enter to toggle)")
    print("(Ctrl-C also exits)\n")

    with serial.Serial(port, baud, timeout=1) as ser:
        while not stop:
            # ── drain any pending keyboard commands ───────────────────────────
            try:
                while True:
                    cmd = cmd_queue.get_nowait()
                    if cmd in ("start", "s"):
                        begin("keyboard")
                    elif cmd in ("stop", "x"):
                        end("keyboard")
                    elif cmd in ("quit", "q"):
                        stop = True
                    elif cmd == "":
                        # Bare Enter toggles recording
                        (end if recording else begin)("keyboard")
                    else:
                        print(f"  [?] Unknown command '{cmd}'. "
                              f"Use: start | stop | quit  (Enter toggles)")
            except queue.Empty:
                pass
            if stop:
                break

            raw = ser.readline()
            if not raw:
                continue  # 1-second readline timeout — loop to recheck commands

            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            # ── control sentinels from firmware ───────────────────────────────
            if "START_RECORDING" in line:
                begin("firmware")
                continue

            if "STOP_RECORDING" in line:
                end("firmware")
                continue

            # ── data lines (only captured while recording) ────────────────────
            if not recording:
                # Show non-recording firmware output without cluttering the terminal
                print(f"  [idle] {line}")
                continue

            row = parse_line(line)
            if row is None:
                # Non-data line during recording (e.g. firmware debug print)
                print(f"  [skip] {line}")
                continue

            row["timestamp"] = datetime.now()
            rows.append(row)
            print(
                f"  {row['timestamp'].strftime('%H:%M:%S.%f')[:-3]}  "
                f"PM2.5={row['pm2_5']:.2f}  Temp={row['temp']:.1f}  "
                f"RH={row['rh']:.1f}  VOC={row['voc']:.0f}  NOx={row['nox']:.0f}"
            )

    # ── exit: flush any in-progress session ───────────────────────────────────
    if recording and rows:
        print("\nExiting — saving in-progress session…")
        save_session(rows, out_dir)
    elif not rows:
        print("\nNo data captured.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log SEN55 UART data to timestamped .pkl files, controlled "
                    "by keyboard commands or START_RECORDING / STOP_RECORDING "
                    "sentinels."
    )
    parser.add_argument(
        "--port", required=True,
        help="Serial port, e.g. /dev/ttyUSB0 or COM3"
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "--out", default="./data",
        help="Output directory for .pkl files (default: ./data)"
    )
    args = parser.parse_args()

    collect(
        port=args.port,
        baud=args.baud,
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
