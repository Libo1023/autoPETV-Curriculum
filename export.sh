#!/bin/bash
set -euo pipefail
echo "Exporting Docker image to autopetv_submission.tar.gz ..."
docker save autopetv_submission | gzip -c > autopetv_submission.tar.gz
echo "Done. File: autopetv_submission.tar.gz ($(du -h autopetv_submission.tar.gz | cut -f1))"