#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config
# -----------------------------
NP=8
BIN=./bin/ex2p
ORDER=3
MTH=0         # 0 = Neumann (big-domain)
RES=1         # 1 = print "L2 error" (residual mode)

RUNS=20
RESULTS_DIR="results"
SUMMARY="${RESULTS_DIR}/ex2p_big_domain_summary.tsv"

# -----------------------------
# Mesh list and their b/a values
# -----------------------------
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

# -----------------------------
# Prep
# -----------------------------
mkdir -p "${RESULTS_DIR}"
echo -e "# b_over_a\tmean_relL2\tmean_solve_time[s]\tmean_assembly_time[s]" > "${SUMMARY}"

# Sanity checks
if [[ ! -x "${BIN}" ]]; then
  echo "Error: binary not found or not executable: ${BIN}" >&2
  exit 1
fi
if [[ ${#MESHES[@]} -ne ${#BA_VALUES[@]} ]]; then
  echo "Error: MESHES and BA_VALUES length mismatch." >&2
  exit 1
fi

# -----------------------------
# Run loops
# -----------------------------
for idx in "${!MESHES[@]}"; do
  MESH="${MESHES[$idx]}"
  BA="${BA_VALUES[$idx]}"
  TAG=$(basename "$MESH" .msh)

  if [[ ! -f "$MESH" ]]; then
    echo "Warning: mesh not found: $MESH — skipping." >&2
    continue
  fi

  echo ">> Running $TAG (b/a=$BA) ${RUNS} times..."

  rel_sum=0.0
  solve_sum=0.0
  assembly_sum=0.0
  ok=0

  for r in $(seq 1 "${RUNS}"); do
    echo "   - run $r/${RUNS}"
    LOG="${RESULTS_DIR}/${TAG}.run${r}.log"

    # Note: if your cluster prepends rank tags to stdout lines, the relaxed regex below still matches.
    mpirun -np "${NP}" "${BIN}" -mth "${MTH}" -m "${MESH}" -o "${ORDER}" -res "${RES}" > "${LOG}" 2>&1

    # Parse metrics from program output (rank 0 prints):
    #   "L2 error: <val>"
    #   "Solver time: <val> s"
    #   "Assembly time: <val> s"
    rel=$(awk '/L2 error:[[:space:]]*/{print $3; found=1} END{if(!found) exit 1}' "$LOG" || echo "NaN")
    solve=$(awk '/Solver time:[[:space:]]*/{print $3; found=1} END{if(!found) exit 1}' "$LOG" || echo "NaN")
    assem=$(awk '/Assembly time:[[:space:]]*/{print $3; found=1} END{if(!found) exit 1}' "$LOG" || echo "NaN")

    if [[ "$rel" == "NaN" || "$solve" == "NaN" || "$assem" == "NaN" ]]; then
      echo "     ! Could not parse metrics for $TAG run $r — check $LOG" >&2
      continue
    fi

    rel_sum=$(awk -v a="$rel_sum" -v b="$rel"     'BEGIN{printf "%.16f", a+b}')
    solve_sum=$(awk -v a="$solve_sum" -v b="$solve" 'BEGIN{printf "%.16f", a+b}')
    assembly_sum=$(awk -v a="$assembly_sum" -v b="$assem" 'BEGIN{printf "%.16f", a+b}')
    ((ok++))
  done

  if (( ok == 0 )); then
    echo -e "${BA}\tNaN\tNaN\tNaN" | tee -a "${SUMMARY}" > /dev/null
    continue
  fi

  rel_mean=$(awk -v s="$rel_sum" -v n="$ok" 'BEGIN{printf "%.10e", s/n}')
  solve_mean=$(awk -v s="$solve_sum" -v n="$ok" 'BEGIN{printf "%.6f", s/n}')
  assembly_mean=$(awk -v s="$assembly_sum" -v n="$ok" 'BEGIN{printf "%.6f", s/n}')

  echo -e "${BA}\t${rel_mean}\t${solve_mean}\t${assembly_mean}" | tee -a "${SUMMARY}" > /dev/null
done

echo "Done. Summary written to: ${SUMMARY}"

