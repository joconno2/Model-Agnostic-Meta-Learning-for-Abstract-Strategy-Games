#!/usr/bin/env bash
# Publish MAML results to GitHub so they're visible from anywhere.
# Intended to run as a cron job or after training finishes.
#
# Commits runs/game_task_v1/loss.png + a RESULTS.md summary to the repo.

set -euo pipefail
cd "$(dirname "$0")"

RUN_DIR="./runs/game_task_v1"
RESULTS_MD="RESULTS.md"

# Only run if there's something to publish
if [[ ! -f "$RUN_DIR/loss.png" ]] && [[ ! -f "$RUN_DIR/config.txt" ]]; then
    echo "No results yet."
    exit 0
fi

# Generate summary
source .venv/bin/activate 2>/dev/null || true

python3 -c "
import sys, os
from pathlib import Path

run_dir = Path('$RUN_DIR')
lines = ['# MAML Value-Only ANIL — Results', '']
lines.append('> Auto-generated. Updated each validation step.')
lines.append('')

# Config
config_path = run_dir / 'config.txt'
if config_path.exists():
    lines.append('## Config')
    lines.append('\`\`\`')
    lines.extend(config_path.read_text().splitlines())
    lines.append('\`\`\`')
    lines.append('')

# Checkpoint stats
try:
    import torch
    for name in ['latest.pt', 'best.pt']:
        p = run_dir / name
        if p.exists():
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            it = ckpt.get('iteration', '?')
            best = ckpt.get('best_val_meta', '?')
            train = ckpt.get('train_meta_history', [])
            val = ckpt.get('val_meta_history', [])
            lines.append(f'## {name}')
            lines.append(f'- Iteration: {it}')
            if isinstance(best, float):
                lines.append(f'- Best val meta-loss: {best:.4f}')
            if train:
                lines.append(f'- Latest train meta-loss: {train[-1]:.4f}')
                lines.append(f'- Train loss range (last 50): {min(train[-50:]):.4f} – {max(train[-50:]):.4f}')
            if val:
                lines.append(f'- Latest val meta-loss: {val[-1]:.4f}')
                lines.append(f'- Val loss range (last 10): {min(val[-10:]):.4f} – {max(val[-10:]):.4f}')
            lines.append('')
except ImportError:
    lines.append('(torch not available — checkpoint stats skipped)')
    lines.append('')

# Loss plot
if (run_dir / 'loss.png').exists():
    lines.append('## Loss Curve')
    lines.append('![Loss](runs/game_task_v1/loss.png)')
    lines.append('')

# Console log tail
console = Path('./runs/game_task_v1_console.log')
if console.exists():
    log_lines = console.read_text().splitlines()
    lines.append(f'## Console Log (last 30 of {len(log_lines)} lines)')
    lines.append('\`\`\`')
    lines.extend(log_lines[-30:])
    lines.append('\`\`\`')

Path('$RESULTS_MD').write_text(chr(10).join(lines) + chr(10))
print('Generated $RESULTS_MD')
"

# Stage and commit
git add -f "$RESULTS_MD" 2>/dev/null || true
git add -f "$RUN_DIR/loss.png" 2>/dev/null || true
git add -f "$RUN_DIR/config.txt" 2>/dev/null || true

if git diff --cached --quiet; then
    echo "No changes to publish."
    exit 0
fi

git commit -m "results: auto-update $(date '+%Y-%m-%d %H:%M')" --no-verify 2>/dev/null
git push origin main 2>/dev/null
echo "Published to GitHub at $(date)"
