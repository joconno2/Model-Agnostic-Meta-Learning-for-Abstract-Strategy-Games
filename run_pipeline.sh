#!/bin/bash
# Run the full Stockfish preprocessing pipeline on threadripper.
# Expects stockfish_eval.py to have already completed.
set -e

cd ~/maml-dasg
VENV=.venv/bin/python

echo "=== Step 1: Extract openings DB ==="
$VENV extract_openings.py --db lichess_chess.sqlite --out sf_openings.sqlite

echo "=== Step 2: Verify Stockfish coverage ==="
TOTAL=$($VENV -c "
import sqlite3
c = sqlite3.connect('lichess_chess.sqlite')
total = c.execute('SELECT COUNT(*) FROM positions').fetchone()[0]
done = c.execute('SELECT COUNT(*) FROM positions WHERE stockfish_cp IS NOT NULL').fetchone()[0]
print(f'{done}/{total} ({100*done/total:.1f}%)')
c.close()
")
echo "Stockfish coverage: $TOTAL"

echo "=== Step 3: Preprocess to npz with Stockfish targets ==="
mkdir -p processed_sf_chess
$VENV db_preprocess_chess_sf.py lichess_chess.sqlite processed_sf_chess

echo "=== Step 4: Count shards ==="
ls processed_sf_chess/*.npz | wc -l
echo "shards written"

echo "=== Pipeline complete ==="
echo "Next: transfer processed_sf_chess/ and sf_openings.sqlite to DGX"
echo "Then run train_disjoint.py with --data-dir processed_sf_chess --db-path sf_openings.sqlite"
