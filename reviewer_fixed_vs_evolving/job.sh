#!/bin/bash
set -euo pipefail
root="$1"
shift
export PYTHONDONTWRITEBYTECODE=1
cd "$root/source"
exec "$root/env/bin/python" -B -u -m reviewer_fixed_vs_evolving.job --root "$root" "$@"
