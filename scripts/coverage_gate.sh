#!/usr/bin/env bash
set -euo pipefail

# Enforce coverage thresholds using cargo-llvm-cov LCOV output.
#
# Requirements:
#   - cargo-llvm-cov installed: `cargo install cargo-llvm-cov`
#   - llvm-tools component: `rustup component add llvm-tools-preview`
#
# Usage:
#   scripts/coverage_gate.sh            # uses defaults: file>=70, project>=85
#   FILE_THRESHOLD=75 PROJECT_THRESHOLD=88 scripts/coverage_gate.sh

FILE_THRESHOLD="${FILE_THRESHOLD:-70}"
PROJECT_THRESHOLD="${PROJECT_THRESHOLD:-85}"

echo "[coverage] thresholds: file>=${FILE_THRESHOLD}% project>=${PROJECT_THRESHOLD}%"

out_dir="target/llvm-cov"
lcov_file="coverage.lcov"

mkdir -p "${out_dir}"

echo "[coverage] running cargo llvm-cov to produce LCOV..."
cargo llvm-cov --workspace --all-features --lcov --output-path "${lcov_file}" > /dev/null

if [[ ! -s "${lcov_file}" ]]; then
  echo "[coverage] ERROR: LCOV file not produced: ${lcov_file}" >&2
  exit 2
fi

echo "[coverage] parsing LCOV for per-file and overall coverage..."

# Parse LCOV to compute per-file and overall percentages.
overall_total=0
overall_covered=0

failed_files=()

awk -v thres="${FILE_THRESHOLD}" '
  BEGIN { FS="[:=,]" }
  /^SF:/ { if (file!="") {
              pct=(covered>0? (covered*100.0/total):0);
              printf "%s\t%.2f\t%d\t%d\n", file, pct, covered, total;
           }
           file=$0; sub(/^SF:/, "", file); covered=0; total=0; next }
  /^DA:/ { if ($3+0>0) covered++; total++; next }
  END {
      if (file!="") {
          pct=(covered>0? (covered*100.0/total):0);
          printf "%s\t%.2f\t%d\t%d\n", file, pct, covered, total;
      }
  }
' "${lcov_file}" | while IFS=$'\t' read -r file pct cov tot; do
  # Track overall
  overall_total=$((overall_total + tot))
  overall_covered=$((overall_covered + cov))

  # Only gate source files under src/ (ignore tests/ and generated paths)
  if [[ "$file" == *"/src/"* ]]; then
    ipct_int=${pct%.*}
    if (( ipct_int < FILE_THRESHOLD )); then
      failed_files+=("$file ($pct%)")
    fi
  fi
done

project_pct=0
if (( overall_total > 0 )); then
  project_pct=$(( overall_covered * 100 / overall_total ))
fi

echo "[coverage] project lines covered: ${overall_covered}/${overall_total} (~${project_pct}%)"

status=0
if (( project_pct < PROJECT_THRESHOLD )); then
  echo "[coverage] ERROR: project coverage ${project_pct}% < ${PROJECT_THRESHOLD}%" >&2
  status=3
fi

if (( ${#failed_files[@]} > 0 )); then
  echo "[coverage] ERROR: files below ${FILE_THRESHOLD}%:"
  for f in "${failed_files[@]}"; do echo "  - $f"; done
  status=4
fi

if (( status == 0 )); then
  echo "[coverage] OK: thresholds met"
fi

exit "$status"

