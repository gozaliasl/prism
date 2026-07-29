#!/usr/bin/env bash
# fetch_tng_particles.sh
# Wrapper for batch_fetch_galaxygenius_stamps.py that loads TNG_API_KEY from
# a local, gitignored credentials file instead of pasting it on the command
# line each time.
#
# One-time setup:
#   echo 'TNG_API_KEY=<your key>' > .tng_api_key
#   chmod 600 .tng_api_key
#
# Usage:
#   ./scripts/local/fetch_tng_particles.sh <candidates.csv> [output_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

KEY_FILE="$PROJECT_ROOT/.tng_api_key"
if [[ -z "${TNG_API_KEY:-}" ]]; then
    if [[ -f "$KEY_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$KEY_FILE"
    else
        echo "ERROR: TNG_API_KEY not set and $KEY_FILE not found." >&2
        echo "       Create it with: echo 'TNG_API_KEY=<your key>' > $KEY_FILE" >&2
        exit 1
    fi
fi

CANDIDATES="${1:?Usage: $0 <candidates.csv> [output_dir]}"
OUTPUT_DIR="${2:-/Volumes/extHD/galaxygenius_build/workspace/data}"

PYTHON="/Users/gozalig1/anaconda3/envs/astro-clean/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python3"

TNG_API_KEY="$TNG_API_KEY" "$PYTHON" scripts/batch_fetch_galaxygenius_stamps.py \
    --candidates "$CANDIDATES" \
    --output-dir "$OUTPUT_DIR"
