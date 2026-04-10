#!/usr/bin/env python3
"""
Distribute processed chess data to Ray cluster workers.

Pushes the processed_chess_flat/ npz shards (and optionally the SQLite DB)
to a known path on each worker so MAMLActor can load them locally.

Usage:
    python distribute_data.py \
        --data-dir ./processed_chess_flat \
        --remote-dir /tmp/maml-chess/processed_chess_flat

    # With SQLite DB (for opening/player task mode):
    python distribute_data.py \
        --data-dir ./processed_chess_flat \
        --db-path ./lichess_2016-03_elo1800_base300.sqlite \
        --remote-dir /tmp/maml-chess/processed_chess_flat \
        --remote-db-dir /tmp/maml-chess
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import paramiko

sys.path.insert(0, str(Path(__file__).parent / ".." / "DragonchessAI-Research"))
from cluster.worker_config import load_workers_csv

CAMELRAY_WORKERS = Path.home() / "research" / "camelRay" / "workers.csv"


def push_to_worker(host, user, password, local_dir, remote_dir, db_path=None, remote_db_dir=None):
    """SFTP the data directory to one worker."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(
            host, username=user, password=password,
            allow_agent=False, look_for_keys=False,
            timeout=15, auth_timeout=15, banner_timeout=15,
        )
    except Exception as e:
        return f"SSH_FAIL: {e}"

    try:
        # mkdir -p remote_dir
        ssh.exec_command(f"mkdir -p {remote_dir}", timeout=10)[1].channel.recv_exit_status()

        sftp = ssh.open_sftp()

        # Push all .npz files
        local_path = Path(local_dir)
        files = sorted(local_path.glob("*.npz"))
        for f in files:
            remote_path = f"{remote_dir}/{f.name}"
            sftp.put(str(f), remote_path)

        # Push SQLite DB if provided
        if db_path and remote_db_dir:
            ssh.exec_command(f"mkdir -p {remote_db_dir}", timeout=10)[1].channel.recv_exit_status()
            db_name = Path(db_path).name
            sftp.put(str(db_path), f"{remote_db_dir}/{db_name}")

        sftp.close()
        return f"OK ({len(files)} shards)"
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        try:
            ssh.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Distribute chess data to workers")
    parser.add_argument("--data-dir", required=True, help="Local path to processed_chess_flat/")
    parser.add_argument("--db-path", default=None, help="Local path to SQLite DB (optional)")
    parser.add_argument("--remote-dir", default="/tmp/maml-chess/processed_chess_flat")
    parser.add_argument("--remote-db-dir", default="/tmp/maml-chess")
    parser.add_argument("--workers-file", default=str(CAMELRAY_WORKERS))
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()

    local_dir = Path(args.data_dir)
    if not local_dir.is_dir():
        print(f"ERROR: {local_dir} is not a directory")
        sys.exit(1)

    npz_files = sorted(local_dir.glob("*.npz"))
    total_mb = sum(f.stat().st_size for f in npz_files) / 1e6
    print(f"Data: {len(npz_files)} shards, {total_mb:.1f} MB total")

    if args.db_path:
        db_mb = Path(args.db_path).stat().st_size / 1e6
        print(f"DB: {args.db_path} ({db_mb:.1f} MB)")

    workers = load_workers_csv(args.workers_file, include_disabled=False)
    print(f"Workers: {len(workers)} enabled in {args.workers_file}")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for w in workers:
            f = pool.submit(
                push_to_worker,
                w.ssh_host, w.username, w.password,
                str(local_dir), args.remote_dir,
                args.db_path, args.remote_db_dir,
            )
            futures[f] = w.display_name

        ok = 0
        fail = 0
        for f in as_completed(futures):
            name = futures[f]
            result = f.result()
            status = "OK" if result.startswith("OK") else "FAIL"
            if status == "OK":
                ok += 1
            else:
                fail += 1
            print(f"  {name:<18} {result}")

    print(f"\n{ok} succeeded, {fail} failed")


if __name__ == "__main__":
    main()
