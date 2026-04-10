#!/usr/bin/env python3
"""
Email a MAML training summary. Run manually or from a cron job.

Usage:
    python notify_results.py ./runs/game_task_v1

Reads latest.pt checkpoint and console log, composes a summary, and emails
it via Gmail SMTP. Credentials from environment or camelRay/.env.
"""

import json
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path


def load_env():
    """Load SMTP creds. Try env vars first, then camelRay/.env for the app password."""
    smtp_user = os.environ.get("SMTP_USER", "jim@oconnor-computing.com")
    smtp_pass = os.environ.get("SMTP_PASS", "")

    if not smtp_pass:
        env_file = Path.home() / "research" / "assistant" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("GMAIL_PASSWORD_1="):
                    smtp_pass = line.split("=", 1)[1].strip()

    return smtp_user, smtp_pass


def build_summary(run_dir: Path) -> str:
    lines = []
    lines.append(f"MAML Run: {run_dir.name}")
    lines.append("=" * 50)

    # Config
    config_path = run_dir / "config.txt"
    if config_path.exists():
        lines.append("\nConfig:")
        for line in config_path.read_text().splitlines()[:20]:
            lines.append(f"  {line}")

    # Checkpoint
    import torch
    for ckpt_name in ["latest.pt", "best.pt"]:
        ckpt_path = run_dir / ckpt_name
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            iteration = ckpt.get("iteration", "?")
            best_val = ckpt.get("best_val_meta", "?")
            train_hist = ckpt.get("train_meta_history", [])
            val_hist = ckpt.get("val_meta_history", [])

            lines.append(f"\n{ckpt_name}:")
            lines.append(f"  Iteration: {iteration}")
            lines.append(f"  Best val meta-loss: {best_val}")
            if train_hist:
                lines.append(f"  Last 10 train losses: {[round(x, 4) for x in train_hist[-10:]]}")
            if val_hist:
                lines.append(f"  Last 5 val losses: {[round(x, 4) for x in val_hist[-5:]]}")

    # Console log tail
    console_log = run_dir.parent / f"{run_dir.name}_console.log"
    if console_log.exists():
        log_lines = console_log.read_text().splitlines()
        lines.append(f"\nConsole log ({len(log_lines)} lines total), last 20:")
        for l in log_lines[-20:]:
            lines.append(f"  {l}")

    # Loss plot
    loss_png = run_dir / "loss.png"
    lines.append(f"\nLoss plot: {'exists' if loss_png.exists() else 'not yet'}")

    return "\n".join(lines)


def send_email(subject: str, body: str, to: str, smtp_user: str, smtp_pass: str,
               attachment_path: Path = None):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to
    msg.set_content(body)

    if attachment_path and attachment_path.exists():
        with open(attachment_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="image",
                subtype="png",
                filename=attachment_path.name,
            )

    with smtplib.SMTP("smtp.gmail.com", 587) as s:
        s.starttls()
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)
    print(f"Email sent to {to}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_dir> [recipient_email]")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    to = sys.argv[2] if len(sys.argv) > 2 else "joconno2@conncoll.edu"

    smtp_user, smtp_pass = load_env()
    if not smtp_pass:
        print("No SMTP password found. Set SMTP_PASS env var or ensure ~/research/assistant/.env has GMAIL_PASSWORD_1.")
        # Still print the summary even if we can't email
        print(build_summary(run_dir))
        sys.exit(1)

    summary = build_summary(run_dir)
    print(summary)

    loss_png = run_dir / "loss.png"
    send_email(
        subject=f"[MAML] {run_dir.name} — status update",
        body=summary,
        to=to,
        smtp_user=smtp_user,
        smtp_pass=smtp_pass,
        attachment_path=loss_png if loss_png.exists() else None,
    )


if __name__ == "__main__":
    main()
