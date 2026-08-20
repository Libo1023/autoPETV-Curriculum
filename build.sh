#!/bin/bash
set -euo pipefail
SCRIPTPATH="$(cd "$(dirname "$0")" && pwd)"
echo "Building Docker image: autopetv_submission"
docker build -t autopetv_submission "$SCRIPTPATH"
echo "Done."