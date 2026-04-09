"""
uart_logger.py — Reads SEN55 data from serial UART, stores it in a
timestamped pandas DataFrame, and saves it as a .pkl file.

Recording is controlled by sentinel strings sent over UART:
  START_RECORDING  — firmware signals that a session has begun
  STOP_RECORDING   — firmware signals that the session has ended

Each START/STOP pair produces one .pkl file. Multiple pairs in the same
run each produce their own file (the output filename is timestamped at
the moment START_RECORDING is received).

Dependencies: pyserial, pandas  (pip install pyserial pandas)

Usage:
    python uart_logger.py --port /dev/ttyUSB0 --baud 115200 --out ./data
    python uart_logger.py --port COM3 --out ./data

Expected serial line format (from firmware):
    [SEN55] PM1=1.23 PM2.5=2.34 PM4=3.45 PM10=4.56 Temp=25.1 RH=50.2 VOC=100.0 NOx=10.0
"""

import argparse
import re
import signal
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


def collect(port: str, baud: int, out_dir: Path) -> None:
    """
    Keep the serial port open indefinitely, waiting for START_RECORDING /
    STOP_RECORDING sentinels from the firmware.

    State machine:
        WAITING  →  (START_RECORDING)  →  RECORDING
        RECORDING →  (STOP_RECORDING)  →  WAITING   (file written here)

    Ctrl-C flushes any in-progress session before exiting.
    """
    recording = False   # True while between START_RECORDING and STOP_RECORDING
    rows: list[dict] = []
    stop = False        # Set to True by Ctrl-C

    def _on_sigint(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    print(f"Listening on {port} at {baud} baud.  Waiting for START_RECORDING…")
    print("(Ctrl-C to exit)\n")

    with serial.Serial(port, baud, timeout=1) as ser:
        while not stop:
            raw = ser.readline()
            if not raw:
                continue  # 1-second readline timeout — keep waiting

            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            # ── control sentinels ─────────────────────────────────────────────
            if "START_RECORDING" in line:
                if recording:
                    # Firmware sent START again without STOP — save what we have
                    print("  [warn] START_RECORDING received while already recording.")
                    print("  Saving current session and starting a new one.")
                    save_session(rows, out_dir)
                    rows = []
                recording = True
                print(f"  [START] Recording began at {datetime.now().strftime('%H:%M:%S')}")
                continue

            if "STOP_RECORDING" in line:
                if not recording:
                    print("  [warn] STOP_RECORDING received but not currently recording — ignored.")
                    continue
                recording = False
                print(f"  [STOP]  Recording ended at {datetime.now().strftime('%H:%M:%S')}")
                save_session(rows, out_dir)
                rows = []
                print("\nWaiting for next START_RECORDING…")
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

    # ── Ctrl-C: flush any in-progress session ─────────────────────────────────
    if recording and rows:
        print("\nInterrupted — saving partial session…")
        save_session(rows, out_dir)
    elif not rows:
        print("\nNo data captured.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log SEN55 UART data to timestamped .pkl files, "
                    "gated by START_RECORDING / STOP_RECORDING sentinels."
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
