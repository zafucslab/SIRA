#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/train.sh" tonglu "${1:-2021}"
