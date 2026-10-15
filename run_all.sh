#!/bin/sh
# Run every self-check in this package against the case-level tables.
#
#   ./run_all.sh /path/to/case_level_tables.xlsx [--ablation /path/to/ablation_record.xlsx]
#
# Each script prints one line per quantity: the value it computed and the value
# printed in the manuscript. A mismatch exits non-zero, and the shell stops, so
# the run as a whole reports failure if any single check does.
set -e
DATA="${1:?usage: ./run_all.sh PATH_TO_case_level_tables.xlsx [--ablation PATH]}"
shift

HERE=$(dirname "$0")
PY=${PY:-python3}

echo "############  reproduce.py  (Tiers 1-3, measurement reliability, ablation)"
"$PY" "$HERE/reproduce.py" --data "$DATA" "$@"

echo
echo "############  comparators.py  (Table 4, baselines and information-matched models)"
"$PY" "$HERE/comparators.py" --data "$DATA"

echo
echo "############  surgeon_comparison.py  (Table 4, three independent surgeon estimates)"
"$PY" "$HERE/surgeon_comparison.py" --data "$DATA"

echo
echo "############  sampling.py  (how the Tier 2 and Tier 3A cases were drawn)"
"$PY" "$HERE/sampling.py" --data "$DATA"

echo
echo "############  preservation.py  (regional registration comparison)"
"$PY" "$HERE/preservation.py"

echo
echo "all checks passed"
