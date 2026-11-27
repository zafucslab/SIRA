#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/train.sh" longyou "${1:-2021}"
