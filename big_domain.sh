#!/usr/bin/env bash
set -euo pipefail

# Config
NP=8
BIN=./bin/Poisson_big_domain
ORDER=3
BC=0   # 0 = Dirichlet(0), 1 = Neumann(0)  <-- force Neumann

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

# Compute b/a numbers (numeric) to go alongside the above meshes
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
SUMMARY="results/big_domain_summary.tsv"
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

# Run each mesh 20 times and average metrics
for idx in "${!MESHES[@]}"; do
  MESH="${MESHES[$idx]}"
  BA="${BA_VALUES[$idx]}"
  TAG=$(basename "$MESH" .msh)

  if [[ ! -f "$MESH" ]]; then
    echo "Warning: mesh not found: $MESH — skipping." >&2
    continue
  fi

  echo ">> Running $TAG (b/a=$BA) with Neumann(0) (-bc ${BC}) 20 times..."

  rel_sum=0.0
  solve_sum=0.0
  runs=20

  for r in $(seq 1 $runs); do
    echo "   - run $r/$runs"
    mpirun -np "$NP" "$BIN" \
      -m "$MESH" \
      -o "$ORDER" \
      -bc "$BC" \
      > "results/${TAG}.run${r}.log" 2>&1

    # Parse metrics immediately (files get overwritten each run, so read now)
    REL_FILE="results/${TAG}.sm.relL2.txt"
    TIM_FILE="results/${TAG}.timings.txt"

    if [[ ! -f "$REL_FILE" || ! -f "$TIM_FILE" ]]; then
      echo "     ! Missing results for $TAG on run $r — check logs: results/${TAG}.run${r}.log" >&2
      continue
    fi

    rel=$(awk '{print $2}' "$REL_FILE")
    solve=$(awk '/^solve[[:space:]]/{print $2}' "$TIM_FILE")

    # Accumulate with awk to keep good precision
    rel_sum=$(awk -v a="$rel_sum" -v b="$rel" 'BEGIN{printf "%.16f", a+b}')
    solve_sum=$(awk -v a="$solve_sum" -v b="$solve" 'BEGIN{printf "%.16f", a+b}')
  done

  # Means
  rel_mean=$(awk -v s="$rel_sum" -v n="$runs" 'BEGIN{printf "%.10e", s/n}')
  solve_mean=$(awk -v s="$solve_sum" -v n="$runs" 'BEGIN{printf "%.6f", s/n}')

  echo -e "${BA}\t${rel_mean}\t${solve_mean}" | tee -a "$SUMMARY" > /dev/null
done

echo "Done. Summary written to: $SUMMARY"

