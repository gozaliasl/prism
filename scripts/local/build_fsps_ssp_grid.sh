#!/usr/bin/env bash
# Fetch the FSPS data checkout (isochrones + spectral libraries, ~1.3 GB)
# needed for SPS_HOME -- the pip-installed `fsps` package only ships the
# Fortran source/bindings, not this data. One-time download.
set -euo pipefail

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data/fsps_home"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

if [ ! -d "fsps-master" ]; then
  echo "Downloading FSPS data checkout..."
  curl -L -o fsps.tar.gz https://codeload.github.com/cconroy20/fsps/tar.gz/refs/heads/master
  tar xzf fsps.tar.gz
  rm fsps.tar.gz
fi

echo "SPS_HOME=$DEST_DIR/fsps-master"
echo "Run: SPS_HOME=$DEST_DIR/fsps-master python scripts/local/build_fsps_ssp_grid.py"
