#!/usr/bin/env bash
set -euo pipefail

# ------------------------
# Configuration
# ------------------------
NP=8
BIN=./bin/ex2p_ext
MESH=mesh/simple_3d_ref.msh
ORDER=3
METHOD=1
LIN=0
LVALS=(0 2 4 8 16 32)
NRUN=20

# ------------------------
# Output setup
# ------------------------
OUTDIR=results
RAW_DIR="${OUTDIR}/raw"
SUMMARY_FILE="${OUTDIR}/DtN.txt"

mkdir -p "${RAW_DIR}"

# Header line
echo -e "# lmax\tassembly_time\tsolver_time\tL2_error" > "${SUMMARY_FILE}"

# ------------------------
# Main loop
# ------------------------
for L in "${LVALS[@]}"; do
  RAW_FILE="${RAW_DIR}/run_lmax_${L}.txt"
  : > "${RAW_FILE}"

  echo "==> Running lmax=${L} (${NRUN} trials)..."

  for ((i=1; i<=NRUN; i++)); do
    OUT="$(mpirun -np "${NP}" "${BIN}" -m "${MESH}" -o "${ORDER}" -mth "${METHOD}" -lin "${LIN}" -lmax "${L}" 2>&1 || true)"

    ASM="$(awk '/Assembly time:/{print $(NF-1)}' <<< "${OUT}" | tail -n1)"
    SOL="$(awk '/Solver time:/{print $(NF-1)}' <<< "${OUT}" | tail -n1)"
    ERR="$(awk '/L2 error:/{print $NF}'        <<< "${OUT}" | tail -n1)"

    if [[ -z "${ASM}" || -z "${SOL}" || -z "${ERR}" ]]; then
      echo "  [warn] run ${i} for lmax=${L} incomplete, skipping"
      continue
    fi

    echo "${ASM} ${SOL} ${ERR}" >> "${RAW_FILE}"
    echo "  run ${i}/${NRUN}: asm=${ASM}, sol=${SOL}, err=${ERR}"
  done

  # Average over all valid runs
  if [[ ! -s "${RAW_FILE}" ]]; then
    echo "  [error] no valid runs for lmax=${L}; writing NaNs"
    echo -e "${L}\tNaN\tNaN\tNaN" >> "${SUMMARY_FILE}"
    continue
  fi

  read -r AVG_ASM AVG_SOL AVG_ERR < <(
    awk 'BEGIN{asm=0; sol=0; err=0; n=0}
         {asm+=$1; sol+=$2; err+=$3; n++}
         END{if(n>0) printf "%.10g %.10g %.10g\n", asm/n, sol/n, err/n; else print "NaN NaN NaN"}' \
      "${RAW_FILE}"
  )

  echo -e "${L}\t${AVG_ASM}\t${AVG_SOL}\t${AVG_ERR}" >> "${SUMMARY_FILE}"
done

echo
echo "✅ Summary saved to: ${SUMMARY_FILE}"
echo "Raw data in: ${RAW_DIR}/run_lmax_<L>.txt"

