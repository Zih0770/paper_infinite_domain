#!/usr/bin/env bash
set -euo pipefail

# Config
NP=8
BIN=./bin/ex2p
ORDER=3
MTH=0         # always big-domain (Neumann)
RES=1         # request L2 error print from the code

# Mesh list and their b/a values
declare -a MESHES=(
  "mesh/simple_3d_ref.msh"
  "mesh/simple_3d_2.msh"
  "mesh/simple_3d_4.msh"
  "mesh/simple_3d_6.msh"
  "mesh/simple_3d_8.msh"
  "mesh/simple_3d_10.msh"
  "mesh/simple_3d_12.msh"
)

# b/a values (numeric) alongside the meshes above
BA_REF=$(awk 'BEGIN{printf "%.10f\n",10/7}')  # ref: b=1.0, a=0.7
declare -a BA_VALUES=(
  "$BA_REF"
  "2"
  "4"
  "6"
  "8"
  "10"
  "12"
)

# Output summary
mkdir -p results
SUMMARY="results/ex2p_big_domain_summary.tsv"
echo -e "# b_over_a\tmean_relL2\tmean_solve_time[s]" > "$SUMMARY"

# Sanity checks
if [[ ! -x "$BIN" ]]; then
  echo "Error: binary not found or not executable: $BIN" >&2
  exit 1
fi
if [[ ${#MESHES[@]} -ne ${#BA_VALUES[@]} ]]; then
  echo "Error: MESHES and BA_VALUES length mismatch." >&2
  exit 1
fi

# Run each mesh 20 times and average metrics (parse from stdout logs)
for idx in "${!MESHES[@]}"; do
  MESH="${MESHES[$idx]}"
  BA="${BA_VALUES[$idx]}"
  TAG=$(basename "$MESH" .msh)

  if [[ ! -f "$MESH" ]]; then
    echo "Warning: mesh not found: $MESH — skipping." >&2
    continue
  fi

  echo ">> Running $TAG (b/a=$BA) 20 times..."

  rel_sum=0.0
  solve_sum=0.0
  runs=20

  for r in $(seq 1 $runs); do
    echo "   - run $r/$runs"
    LOG="results/${TAG}.run${r}.log"
    mpirun -np "$NP" "$BIN" -mth "$MTH" -m "$MESH" -o "$ORDER" -res "$RES" > "$LOG" 2>&1

    # Parse from program output:
    #   "L2 error: <val>"
    #   "Solver time: <val> s"
    rel=$(awk '/^L2 error:[[:space:]]*/{print $3; found=1} END{if(!found) exit 1}' "$LOG" || echo "NaN")
    solve=$(awk '/^Solver time:[[:space:]]*/{print $3; found=1} END{if(!found) exit 1}' "$LOG" || echo "NaN")

    if [[ "$rel" == "NaN" || "$solve" == "NaN" ]]; then
      echo "     ! Could not parse metrics for $TAG run $r — check $LOG" >&2
      continue
    fi

    rel_sum=$(awk -v a="$rel_sum" -v b="$rel"   'BEGIN{printf "%.16f", a+b}')
    solve_sum=$(awk -v a="$solve_sum" -v b="$solve" 'BEGIN{printf "%.16f", a+b}')
  done

  rel_mean=$(awk -v s="$rel_sum" -v n="$runs" 'BEGIN{printf "%.10e", s/n}')
  solve_mean=$(awk -v s="$solve_sum" -v n="$runs" 'BEGIN{printf "%.6f", s/n}')

  echo -e "${BA}\t${rel_mean}\t${solve_mean}" | tee -a "$SUMMARY" > /dev/null
done

echo "Done. Summary written to: $SUMMARY"

