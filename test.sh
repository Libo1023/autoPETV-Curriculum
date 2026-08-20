#!/bin/bash
#
# test.sh
#
# Run the Docker container locally with test data mounted.
# Requires a local GPU and test data in the expected layout.
#
# Usage:
#   bash test.sh /path/to/test/input /path/to/test/output
#
set -euo pipefail

INPUT_DIR="${1:?Usage: bash test.sh INPUT_DIR OUTPUT_DIR}"
OUTPUT_DIR="${2:?Usage: bash test.sh INPUT_DIR OUTPUT_DIR}"

MEM_LIMIT="30g"

echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

docker run --rm \
    --memory="${MEM_LIMIT}" \
    --memory-swap="${MEM_LIMIT}" \
    --network="none" \
    --cap-drop="ALL" \
    --security-opt="no-new-privileges" \
    --shm-size="2g" \
    --gpus="all" \
    -v "${INPUT_DIR}":/input/ \
    -v "${OUTPUT_DIR}":/output/ \
    autopetv_submission

echo ""
echo "Done. Check output in: ${OUTPUT_DIR}"