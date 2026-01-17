#!/bin/bash
set -euo pipefail

source .env
NUM_RUNS=${NUM_RUNS:-8}

for i in $(seq 1 $NUM_RUNS)
do
    echo "===== Run $i started ====="

    if python3 main.py; then
        echo "Run $i succeeded"
    else
        echo "Run $i failed, continuing to next run"
    fi

    echo "===== Run $i completed ====="
    echo
done

echo "All $NUM_RUNS runs completed!"
