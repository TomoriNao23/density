#!/bin/bash
# ============================================================================
# run_density.sbatch — Single-job ICON EOS-80 density pipeline on DKRZ Levante
#
# Processes periods configured in calculate_density.py sequentially.
# 32 dask workers × 4 threads = 128 cores; ~6 GiB per worker.
#
# Usage:  mkdir -p logs && sbatch run_density.sbatch
# ============================================================================
#SBATCH --job-name=dens_icon
#SBATCH --partition=shared
#SBATCH --account=mh0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=8:00:00
#SBATCH --output=logs/density_%j.out
#SBATCH --error=logs/density_%j.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs state

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311

echo "=== ICON density pipeline ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Python:    $(which python)"
echo "Started:   $(date -Iseconds)"
echo ""

# Raise file descriptor limit — zarr + dask open many files concurrently
ulimit -n 524288

python calculate_density.py --workers 32 --threads 4 --memory "6GiB"

echo ""
echo "Finished:  $(date -Iseconds)"
