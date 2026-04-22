"""
data_uploader.py — Select SEN55 recording sessions and push them to the
                   team's shared cloud storage via rclone.

WHAT THIS SCRIPT DOES:
    1. Reads your local recording directory (where uart_logger.py saves .pkl files).
    2. Lists all available sessions with size and date information.
    3. Lets you choose which sessions to upload (all, specific numbers, a date range).
    4. Calls rclone to copy the selected files to the shared remote.

WHY RCLONE INSTEAD OF BITTORRENT:
    BitTorrent is designed for distributing files to many anonymous public peers.
    For a small team syncing sensor data it has two major problems:
      - Every new session needs a new .torrent file — there is no incremental sync.
      - In DHT mode (tracker-free), your data is announced to the public internet.
    rclone is a single binary that works with Google Drive, S3, Dropbox, SFTP,
    and 50+ other backends. It is fast, encrypted, incremental, and free with
    Google Drive (15 GB). All team members connect their rclone to the same
    shared folder; uploads and downloads are asynchronous.

SETUP (one time per machine):
    1. Install rclone: https://rclone.org/install/
       - Windows: download the .exe from the site, add to PATH.
       - Linux/Mac: curl https://rclone.org/install.sh | sudo bash
    2. Run: rclone config
       - New remote → name it "sensa" (must match rclone_remote in data_config.yaml)
       - Storage type: Google Drive (or whichever backend the team uses)
       - Follow the browser auth steps.
    3. Share the Google Drive folder with all team members.
    4. Each team member sets up their own rclone pointing at the same folder.

USAGE:
    python data_uploader.py
    python data_uploader.py --all                  # skip interactive prompt
    python data_uploader.py --dry-run              # show what would be uploaded
    python data_uploader.py --config my_config.yaml
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """
    Load data_config.yaml. Exit with a helpful message if the file is missing
    (the user probably has not run the setup step yet).
    """
    if not config_path.exists():
        example = config_path.parent / "data_config.example.yaml"
        print(f"ERROR: {config_path} not found.")
        print()
        if example.exists():
            print(f"Copy the example and edit it for your machine:")
            print(f"    cp {example.name} data_config.yaml")
        else:
            print("Create a data_config.yaml with keys: recording_dir, rclone_remote, training_data_dir")
        sys.exit(1)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Expand ~ in paths (e.g. ~/sensa-recordings → /home/user/sensa-recordings)
    cfg['recording_dir'] = Path(cfg['recording_dir']).expanduser()
    return cfg


# ── rclone availability check ─────────────────────────────────────────────────

def check_rclone() -> None:
    """
    Verify that rclone is installed and on the system PATH.
    Exit with setup instructions if it is not.
    """
    result = subprocess.run(
        ['rclone', 'version'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ERROR: rclone is not installed or not on PATH.")
        print()
        print("Install rclone from: https://rclone.org/install/")
        print("Then run: rclone config  (to set up your cloud storage remote)")
        sys.exit(1)


# ── Session discovery ─────────────────────────────────────────────────────────

def find_sessions(recording_dir: Path) -> list[dict]:
    """
    Find all SEN55 .pkl recording sessions in the recording directory.

    Returns a list of dicts sorted by modification time (newest last), each with:
        path     — absolute Path to the .pkl file
        name     — filename only
        size_kb  — file size in kilobytes
        mtime    — datetime of last modification (used for display)
    """
    if not recording_dir.exists():
        print(f"ERROR: Recording directory not found: {recording_dir}")
        print("Check the 'recording_dir' value in data_config.yaml.")
        sys.exit(1)

    pkl_files = sorted(recording_dir.glob("sen55_*.pkl"), key=lambda p: p.stat().st_mtime)

    sessions = []
    for path in pkl_files:
        stat = path.stat()
        sessions.append({
            'path':    path,
            'name':    path.name,
            'size_kb': stat.st_size / 1024,
            'mtime':   datetime.fromtimestamp(stat.st_mtime),
        })

    return sessions


# ── Interactive session selector ──────────────────────────────────────────────

def display_sessions(sessions: list[dict]) -> None:
    """Print the list of available sessions in a readable table."""
    if not sessions:
        print("No sen55_*.pkl files found in the recording directory.")
        return

    print()
    col_w = max(len(s['name']) for s in sessions) + 2
    header = f"  {'#':>3}   {'Filename':<{col_w}}  {'Size':>9}  {'Recorded'}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for i, s in enumerate(sessions, start=1):
        print(
            f"  {i:>3}   {s['name']:<{col_w}}  {s['size_kb']:>7.1f} KB  "
            f"{s['mtime'].strftime('%Y-%m-%d  %H:%M')}"
        )
    print()


def select_sessions(sessions: list[dict]) -> list[dict]:
    """
    Interactively prompt the user to select which sessions to upload.

    Accepted input formats:
        a       → all sessions
        1       → session 1 only
        1,3,5   → sessions 1, 3, and 5
        2-5     → sessions 2 through 5
        <Enter> → cancel (exit without uploading)

    Returns the list of selected session dicts.
    """
    print("Select sessions to upload:")
    print("  [a]     — upload all")
    print("  [1,3]   — upload sessions 1 and 3")
    print("  [1-3]   — upload sessions 1 through 3")
    print("  [Enter] — cancel")
    print()

    raw = input("Your choice: ").strip().lower()

    if not raw:
        print("Cancelled.")
        sys.exit(0)

    if raw == 'a':
        return sessions

    selected = []
    try:
        # Support comma-separated list and/or ranges like "1,3-5,7"
        for part in raw.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-', 1)
                for idx in range(int(start), int(end) + 1):
                    selected.append(sessions[idx - 1])
            else:
                selected.append(sessions[int(part) - 1])
    except (ValueError, IndexError):
        print(f"Invalid selection: '{raw}'. Enter numbers between 1 and {len(sessions)}.")
        sys.exit(1)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in selected:
        if s['name'] not in seen:
            seen.add(s['name'])
            unique.append(s)
    return unique


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_sessions(
    sessions: list[dict],
    rclone_remote: str,
    dry_run: bool = False,
) -> None:
    """
    Copy the selected .pkl files to the rclone remote.

    rclone copyto copies one file at a time to a specific remote path.
    We use copyto (not copy) so the destination filename matches the source.

    Args:
        sessions:      List of session dicts from find_sessions().
        rclone_remote: e.g. "sensa:sensa-data/raw"
        dry_run:       If True, print what would happen but do not transfer.
    """
    total_kb = sum(s['size_kb'] for s in sessions)
    print(f"\nUploading {len(sessions)} session(s)  ({total_kb / 1024:.2f} MB total)")
    if dry_run:
        print("  (DRY RUN — no files will be transferred)")
    print()

    for s in sessions:
        remote_path = f"{rclone_remote.rstrip('/')}/{s['name']}"
        print(f"  → {s['name']}  ({s['size_kb']:.1f} KB)")

        if dry_run:
            print(f"    Would run: rclone copyto {s['path']} {remote_path}")
            continue

        cmd = ['rclone', 'copyto', str(s['path']), remote_path, '--progress']
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"  ERROR: rclone failed for {s['name']} (exit code {result.returncode})")
            print("  Check that your rclone remote is configured and you have write access.")
            sys.exit(1)

    if not dry_run:
        print(f"\nDone. {len(sessions)} file(s) uploaded to {rclone_remote}")
        print("Teammates can download with: python data_sync.py")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Resolve paths relative to this script's directory so the script can be
    # called from any working directory.
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Upload SEN55 recording sessions to the team's shared cloud storage."
    )
    parser.add_argument(
        '--config', default=str(script_dir / 'data_config.yaml'),
        help="Path to data_config.yaml (default: tools/data_config.yaml)"
    )
    parser.add_argument(
        '--all', action='store_true',
        help="Upload all sessions without an interactive prompt."
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Show what would be uploaded but do not transfer any files."
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    check_rclone()

    recording_dir  = cfg['recording_dir']
    rclone_remote  = cfg['rclone_remote']

    print(f"Recording directory : {recording_dir}")
    print(f"rclone remote       : {rclone_remote}")

    sessions = find_sessions(recording_dir)

    if not sessions:
        print("\nNo sessions found. Run uart_logger.py first to collect data.")
        sys.exit(0)

    display_sessions(sessions)

    if args.all:
        selected = sessions
    else:
        selected = select_sessions(sessions)

    upload_sessions(selected, rclone_remote, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
