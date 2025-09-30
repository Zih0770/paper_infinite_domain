#!/usr/bin/env bash
set -euo pipefail

########################
# MPI (OpenMPI 5.0.8)
########################
OMPI_BIN="/space/zy296/source/petsc/arch-opt/bin"
OMPI_LIB="/space/zy296/source/petsc/arch-opt/lib"
MPIRUN="${OMPI_BIN}/mpirun"

if [[ ! -x "$MPIRUN" ]]; then
  echo "ERROR: OpenMPI mpirun not found at ${MPIRUN}" >&2
  exit 1
fi

# Ensure the right libs are picked on all ranks
export LD_LIBRARY_PATH="${OMPI_LIB}:${LD_LIBRARY_PATH-}"

# Quick sanity: confirm this is Open MPI
if ! "$MPIRUN" --version 2>/dev/null | head -1 | grep -qi "Open MPI"; then
  echo "ERROR: ${MPIRUN} is not Open MPI." >&2
  exit 1
fi

########################
# Config
########################
NP=8
BIN=./bin/Poisson_big_domain
ORDER=3
BC=0   # 0 = Dirichlet(0), 1 = Neumann(0)

declare -a MESHES=(
  "mesh/simple_3d_ref.msh"
  "mesh/simple_3d_2.msh"
  "mesh/simple_3d_4.msh"
  "mesh/simple_3d_6.msh"
  "mesh/simple_3d_8.msh"
  "mesh/simple_3d_10.msh"
  "mesh/simple_3d_12.msh"
)

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

########################
# Pre-flight checks
########################
echo "[diag] Using mpirun: ${MPIRUN}"
"$MPIRUN" --version | head -1

# Confirm we actually get NP ranks
COUNT=$("$MPIRUN" -x LD_LIBRARY_PATH -np "$NP" hostname | wc -l)
if [[ "$COUNT" != "$NP" ]]; then
  echo "ERROR: Expected $NP ranks, got $COUNT. Check allocation or host setup." >&2
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "Error: binary not found or not executable: $BIN" >&2
  exit 1
fi

if [[ ${#MESHES[@]} -ne ${#BA_VALUES[@]} ]]; then
  echo "Error: MESHES and BA_VALUES length mismatch." >&2
  exit 1
fi

########################
# Output summary
########################
mkdir -p results
SUMMARY="results/big_domain_summary.tsv"
# Added DOF column (constant per mesh/order)
echo -e "# b_over_a\tdofs\tmean_relL2\tmean_solve_time[s]" > "$SUMMARY"

########################
# Runs
########################
for idx in "${!MESHES[@]}"; do
  MESH="${MESHES[$idx]}"
  BA="${BA_VALUES[$idx]}"
  TAG=$(basename "$MESH" .msh)

  if [[ ! -f "$MESH" ]]; then
    echo "Warning: mesh not found: $MESH — skipping." >&2
    continue
  fi

  echo ">> Running $TAG (b/a=$BA) with BC=${BC} (-bc ${BC}) 20 times..."

  rel_sum=0.0
  solve_sum=0.0
  runs=20
  dofs=""

  for r in $(seq 1 $runs); do
    echo "   - run $r/$runs"
    LOG="results/${TAG}.run${r}.log"

    "$MPIRUN" -x LD_LIBRARY_PATH -np "$NP" "$BIN" \
      -m "$MESH" \
      -o "$ORDER" \
      -bc "$BC" \
      > "$LOG" 2>&1

    # Parse metrics immediately (files overwritten each run)
    REL_FILE="results/${TAG}.sm.relL2.txt"
    TIM_FILE="results/${TAG}.timings.txt"

    if [[ ! -f "$REL_FILE" || ! -f "$TIM_FILE" ]]; then
      echo "     ! Missing results for $TAG on run $r — check log: $LOG" >&2
      continue
    fi

    # mean RelL2 and solve time (as before)
    rel=$(awk '{print $2}' "$REL_FILE")
    solve=$(awk '/^solve[[:space:]]/{print $2}' "$TIM_FILE")

    rel_sum=$(awk -v a="$rel_sum" -v b="$rel" 'BEGIN{printf "%.16f", a+b}')
    solve_sum=$(awk -v a="$solve_sum" -v b="$solve" 'BEGIN{printf "%.16f", a+b}')

    # Parse DOFs (once). Your code prints: "Global true dofs: <size>"
    if [[ -z "$dofs" ]]; then
      dofs=$(grep -m1 "Global true dofs:" "$LOG" | awk '{print $4}')
      if [[ -z "${dofs}" ]]; then
        echo "     ! Could not parse DOFs from $LOG (looking for 'Global true dofs: N')." >&2
        dofs="NA"
      fi
    fi
  done

  # Means
  rel_mean=$(awk -v s="$rel_sum" -v n="$runs" 'BEGIN{printf "%.10e", s/n}')
  solve_mean=$(awk -v s="$solve_sum" -v n="$runs" 'BEGIN{printf "%.6f", s/n}')

  # Summary row: add DOFs column
  echo -e "${BA}\t${dofs}\t${rel_mean}\t${solve_mean}" | tee -a "$SUMMARY" > /dev/null
done

echo "Done. Summary written to: $SUMMARY"

