"""
data_sync.py — Pull SEN55 recording sessions from the team's shared cloud
               storage into the local pytorch_calibration training data folder.

WHAT THIS SCRIPT DOES:
    1. Lists all .pkl files available on the rclone remote.
    2. Compares them against what you already have locally.
    3. Shows a summary of new files available.
    4. Downloads the new files (or all files) into training_data_dir.

TYPICAL WORKFLOW:
    Teammate A records data on a lab machine:
        python uart_logger.py --port COM3 --out ~/sensa-recordings
        python data_uploader.py             ← pushes to shared cloud folder

    Teammate B trains the model on a different machine:
        python data_sync.py                 ← pulls new sessions from cloud
        python pytorch_calibration/prepare_data.py
        python pytorch_calibration/main.py

USAGE:
    python data_sync.py
    python data_sync.py --all               # re-download all (overwrite local)
    python data_sync.py --dry-run           # show what would be downloaded
    python data_sync.py --config my.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


# ── Config loading (shared with data_uploader.py) ─────────────────────────────

def load_config(config_path: Path) -> dict:
    """Load the data_sync section of tools/config.yaml.

    Returns just the data_sync sub-dict, so callers see recording_dir,
    rclone_remote and training_data_dir as top-level keys.
    """
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        print("config.yaml is part of the repository — restore it with git.")
        sys.exit(1)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'data_sync' not in cfg:
        print(f"ERROR: {config_path} has no 'data_sync' section.")
        sys.exit(1)

    sync_cfg = cfg['data_sync']
    sync_cfg['recording_dir'] = Path(sync_cfg['recording_dir']).expanduser()
    return sync_cfg


def check_rclone() -> None:
    """Verify rclone is installed. Exit with setup instructions if not."""
    result = subprocess.run(['rclone', 'version'], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: rclone is not installed or not on PATH.")
        print("Install it from: https://rclone.org/install/")
        print("Then run: rclone config  (follow the prompts to add your remote)")
        sys.exit(1)


# ── Remote file listing ───────────────────────────────────────────────────────

def list_remote_sessions(rclone_remote: str) -> list[str]:
    """
    Ask rclone to list all .pkl files on the remote.

    rclone lsf returns one filename per line with no extra formatting.
    We filter to only sen55_*.pkl files so unrelated files in the remote
    folder do not interfere.

    Args:
        rclone_remote: e.g. "sensa:sensa-data/raw"

    Returns:
        Sorted list of filenames (not full paths).
    """
    result = subprocess.run(
        ['rclone', 'lsf', rclone_remote, '--include', 'sen55_*.pkl'],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"ERROR: Could not list files on remote '{rclone_remote}'.")
        print()
        print("Possible causes:")
        print("  - rclone remote is not configured: run 'rclone config'")
        print("  - Remote name is wrong: check 'rclone_remote' under data_sync in tools/config.yaml")
        print("  - No internet connection")
        print()
        print(f"rclone error output:\n{result.stderr}")
        sys.exit(1)

    # lsf outputs "filename\n" per file; strip trailing slashes and blanks.
    files = [line.strip().rstrip('/') for line in result.stdout.splitlines() if line.strip()]
    return sorted(files)


def list_local_sessions(training_data_dir: Path) -> set[str]:
    """
    Return the set of .pkl filenames already present in training_data_dir.

    Args:
        training_data_dir: Absolute path to pytorch_calibration/data/raw/

    Returns:
        Set of filenames (not full paths) already on disk.
    """
    if not training_data_dir.exists():
        return set()
    return {p.name for p in training_data_dir.glob("sen55_*.pkl")}


# ── Display helpers ───────────────────────────────────────────────────────────

def display_status(remote_files: list[str], local_files: set[str]) -> None:
    """
    Print a two-column table showing which remote files are new vs. already local.
    """
    new_files      = [f for f in remote_files if f not in local_files]
    existing_files = [f for f in remote_files if f in local_files]

    print(f"\n  Remote sessions : {len(remote_files)}")
    print(f"  Already local   : {len(existing_files)}")
    print(f"  New (to sync)   : {len(new_files)}")

    if new_files:
        print("\n  New sessions available:")
        for name in new_files:
            print(f"    + {name}")

    if existing_files:
        print(f"\n  Already downloaded ({len(existing_files)} files — skipped unless --all is set):")
        for name in existing_files:
            print(f"    ✓ {name}")

    print()


# ── Download ──────────────────────────────────────────────────────────────────

def download_sessions(
    files_to_download: list[str],
    rclone_remote: str,
    training_data_dir: Path,
    dry_run: bool = False,
) -> None:
    """
    Download the specified files from the rclone remote to training_data_dir.

    We use 'rclone copyto' (file-by-file) rather than 'rclone sync' to avoid
    accidentally deleting local files that are not yet on the remote.

    Args:
        files_to_download: List of filenames to fetch (not full paths).
        rclone_remote:     rclone remote + folder, e.g. "sensa:sensa-data/raw"
        training_data_dir: Local destination directory.
        dry_run:           If True, print commands but do not run them.
    """
    if not files_to_download:
        print("Nothing to download — already up to date.")
        return

    training_data_dir.mkdir(parents=True, exist_ok=True)

    total = len(files_to_download)
    print(f"Downloading {total} file(s) → {training_data_dir}")
    if dry_run:
        print("  (DRY RUN — no files will be transferred)")
    print()

    for name in files_to_download:
        remote_path = f"{rclone_remote.rstrip('/')}/{name}"
        local_path  = training_data_dir / name
        print(f"  ← {name}")

        if dry_run:
            print(f"    Would run: rclone copyto {remote_path} {local_path}")
            continue

        cmd = ['rclone', 'copyto', remote_path, str(local_path), '--progress']
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"  ERROR: rclone failed for {name} (exit code {result.returncode})")
            print("  Check your internet connection and rclone configuration.")
            sys.exit(1)

    if not dry_run:
        print(f"\nDone. {total} file(s) synced to {training_data_dir}")
        print("\nNext steps:")
        print("  python pytorch_calibration/prepare_data.py")
        print("  python pytorch_calibration/main.py")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Pull SEN55 recording sessions from shared cloud storage to your training folder."
    )
    parser.add_argument(
        '--config', default=str(script_dir / 'config.yaml'),
        help="Path to the shared tools/config.yaml (default: tools/config.yaml)"
    )
    parser.add_argument(
        '--all', action='store_true',
        help="Re-download all remote files, including ones already present locally."
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Show what would be downloaded but do not transfer any files."
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    check_rclone()

    rclone_remote     = cfg['rclone_remote']
    # training_data_dir is stored relative to the tools/ directory in the config
    training_data_dir = (script_dir / cfg['training_data_dir']).resolve()

    print(f"rclone remote      : {rclone_remote}")
    print(f"Training data dir  : {training_data_dir}")

    print("\nListing remote sessions ...")
    remote_files = list_remote_sessions(rclone_remote)

    if not remote_files:
        print("No sessions found on the remote.")
        print("A teammate must run data_uploader.py first to share data.")
        sys.exit(0)

    local_files = list_local_sessions(training_data_dir)
    display_status(remote_files, local_files)

    # If --all is set, download everything; otherwise only new files.
    if args.all:
        to_download = remote_files
    else:
        to_download = [f for f in remote_files if f not in local_files]

    download_sessions(to_download, rclone_remote, training_data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
