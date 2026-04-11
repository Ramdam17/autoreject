#!/bin/bash
# ======================================================================
# Pre-download datasets for benchmarks (run on login node — has internet)
#
# Compute nodes have NO internet access. This script caches all datasets
# so the benchmark jobs can run offline.
#
# Usage:
#   bash benchmarks/narval/download_data.sh
# ======================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Pre-downloading benchmark datasets ==="
echo "Repo: $REPO_DIR"
echo ""

# Activate environment
module load StdEnv/2023 python/3.12 scipy-stack cuda/12.2
source "$HOME/envs/autoreject-bench/bin/activate"

cd "$REPO_DIR"

python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- 1. MNE sample dataset ---
logger.info('Downloading MNE sample/testing dataset...')
import mne
try:
    data_path = mne.datasets.testing.data_path(download=True)
    logger.info('  MNE testing data: %s', data_path)
except Exception as e:
    logger.warning('  Testing data failed (%s), trying full sample...', e)
    data_path = mne.datasets.sample.data_path(download=True)
    logger.info('  MNE sample data: %s', data_path)

# --- 2. OpenNeuro ds002778 (Parkinson, 32ch) ---
logger.info('Downloading ds002778 (Parkinson resting-state)...')
from benchmarks.core.datasets import load_ds002778
try:
    epochs, meta = load_ds002778({'subject': 'pd14', 'session': 'off', 'epoch_duration': 3.0})
    logger.info('  ds002778: %d epochs x %d channels', meta['n_epochs'], meta['n_channels'])
except Exception as e:
    logger.error('  ds002778 failed: %s', e)

# --- 3. OpenNeuro ds000117 (Face recognition, ~60ch) ---
logger.info('Downloading ds000117 (Face recognition)...')
from benchmarks.core.datasets import load_ds000117
try:
    epochs, meta = load_ds000117({'subject': '16', 'runs': [3, 4, 5, 6], 'tmin': -0.2, 'tmax': 0.8})
    logger.info('  ds000117: %d epochs x %d channels', meta['n_epochs'], meta['n_channels'])
except Exception as e:
    logger.error('  ds000117 failed: %s', e)

logger.info('All datasets downloaded and cached.')
"

echo ""
echo "=== Data download complete ==="
echo "Datasets cached in:"
echo "  MNE data:  $(python -c 'import mne; print(mne.get_config("MNE_DATA", mne.datasets.utils._get_path(None, None, None)))'  2>/dev/null || echo '$HOME/mne_data')"
echo "  ds002778:  $REPO_DIR/examples/ds002778/"
echo "  ds000117:  $REPO_DIR/examples/ds000117/"
