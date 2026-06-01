#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CXX="${CXX:-nvc++}"
SIZE="${SIZE:-512}"
ITERATIONS="${ITERATIONS:-100}"
EPS="${EPS:-1e-6}"
CHECK_INTERVAL="${CHECK_INTERVAL:-100}"
STAGES=(${STAGES:-1 2 3 4})
RESULT_DIR="${RESULT_DIR:-results/gpu_stage_profiles}"
NSYS_TRACE="${NSYS_TRACE:-cuda,openacc,osrt}"

mkdir -p "$RESULT_DIR/matrices" "$RESULT_DIR/nsys" "$RESULT_DIR/stdout" "$RESULT_DIR/stats" "$RESULT_DIR/openacc"

CSV="$RESULT_DIR/gpu_stage_profiles_${SIZE}_${ITERATIONS}.csv"
TABLE="$RESULT_DIR/gpu_stage_profiles_${SIZE}_${ITERATIONS}.md"

extract_value() {
    local name="$1"
    local file="$2"
    awk -F': ' -v key="$name" '$1 == key {print $2; exit}' "$file" | awk '{print $1}'
}

extract_comment() {
    local file="$1"
    awk -F': ' '$1 == "Comment" {print $2; exit}' "$file"
}

if ! command -v nsys >/dev/null 2>&1; then
    echo "Error: nsys not found. Install NVIDIA Nsight Systems first." >&2
    exit 1
fi

echo "Building GPU binary with CXX=${CXX}"
make CXX="$CXX" therm_gpu

echo "stage,size,max_iterations,eps,time_sec,error,iterations,comment,profile" > "$CSV"

for stage in "${STAGES[@]}"; do
    profile_base="$RESULT_DIR/nsys/gpu_stage_${stage}_${SIZE}_${ITERATIONS}"
    log_file="$RESULT_DIR/gpu_stage_${stage}_${SIZE}_${ITERATIONS}.log"
    profile_stdout_file="$RESULT_DIR/stdout/gpu_stage_${stage}_${SIZE}_${ITERATIONS}_nsys_profile.txt"
    program_output_file="$RESULT_DIR/stdout/gpu_stage_${stage}_${SIZE}_${ITERATIONS}_program.txt"
    stats_file="$RESULT_DIR/stats/gpu_stage_${stage}_${SIZE}_${ITERATIONS}_nsys_stats.txt"
    openacc_stats_file="$RESULT_DIR/openacc/gpu_stage_${stage}_${SIZE}_${ITERATIONS}_openacc_sum.txt"
    matrix_file="$RESULT_DIR/matrices/gpu_stage_${stage}_${SIZE}_${ITERATIONS}.dat"

    echo "Profiling GPU stage ${stage}: size=${SIZE}, iterations=${ITERATIONS}"

    nsys profile --force-overwrite=true --trace="$NSYS_TRACE" -o "$profile_base" \
        ./therm_gpu --size "$SIZE" --eps "$EPS" --iterations "$ITERATIONS" \
        --stage "$stage" --check-interval "$CHECK_INTERVAL" \
        --output "$matrix_file" 2>&1 | tee "$profile_stdout_file" > "$log_file"

    awk '
        /^Stage: / || /^Comment: / || /^Iterations: / || /^Error: / ||
        /^Time: / || /^Saved matrix: / { print }
    ' "$log_file" > "$program_output_file"

    time_sec="$(extract_value "Time" "$log_file")"
    error="$(extract_value "Error" "$log_file")"
    actual_iterations="$(extract_value "Iterations" "$log_file")"
    comment="$(extract_comment "$log_file")"
    profile_file="${profile_base}.nsys-rep"

    if [[ -f "$profile_file" ]]; then
        nsys stats --force-export=true "$profile_file" > "$stats_file" 2>&1 || true
        nsys stats --force-export=true --report openacc_sum --format column \
            "$profile_file" > "$openacc_stats_file" 2>&1 || true
    fi

    echo "${stage},${SIZE},${ITERATIONS},${EPS},${time_sec},${error},${actual_iterations},\"${comment}\",${profile_file}" >> "$CSV"
done

{
    echo "| Этап | Время выполнения, c | Точность | Максимальное количество итераций | Комментарий |"
    echo "|---:|---:|---:|---:|---|"
    tail -n +2 "$CSV" | while IFS=',' read -r stage size max_iterations eps time_sec error iterations comment profile; do
        echo "| ${stage} | ${time_sec} | ${eps} | ${max_iterations} | ${comment//\"/} |"
    done
} > "$TABLE"

echo "Saved CSV: $CSV"
echo "Saved report table: $TABLE"
echo "Saved Nsight reports: $RESULT_DIR/nsys"
echo "Saved nsys profile stdout: $RESULT_DIR/stdout"
echo "Saved nsys stats stdout: $RESULT_DIR/stats"
echo "Saved OpenACC operation summary: $RESULT_DIR/openacc"
