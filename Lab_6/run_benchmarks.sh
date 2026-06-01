#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CXX="${CXX:-nvc++}"
EPS="${EPS:-1e-6}"
ITERATIONS="${ITERATIONS:-1000000}"
REPEATS="${REPEATS:-1}"
CHECK_INTERVAL="${CHECK_INTERVAL:-10}"
RESULT_DIR="${RESULT_DIR:-results}"
PROFILE="${PROFILE:-0}"
PROFILE_ITERATIONS="${PROFILE_ITERATIONS:-100}"

CPU_ONE_SIZES=(${CPU_ONE_SIZES:-128 256 512})
COMMON_SIZES=(${COMMON_SIZES:-128 256 512 1024})
GPU_OPTIMIZATION_SIZE="${GPU_OPTIMIZATION_SIZE:-512}"
GPU_STAGES=(${GPU_STAGES:-1 2 3 4})

mkdir -p "$RESULT_DIR/matrices" "$RESULT_DIR/profiles"

CSV="$RESULT_DIR/benchmarks.csv"
GPU_STAGES_CSV="$RESULT_DIR/gpu_stages.csv"

echo "kind,size,stage,repeat,time_sec,error,iterations,comment" > "$CSV"
echo "stage,size,repeat,time_sec,error,iterations,comment" > "$GPU_STAGES_CSV"

extract_value() {
    local name="$1"
    local file="$2"
    awk -F': ' -v key="$name" '$1 == key {print $2; exit}' "$file" | awk '{print $1}'
}

extract_comment() {
    local file="$1"
    awk -F': ' '$1 == "Comment" {print $2; exit}' "$file"
}

run_case() {
    local kind="$1"
    local binary="$2"
    local size="$3"
    local stage="$4"
    local repeat="$5"
    local log_file="$RESULT_DIR/${kind}_${size}_stage${stage}_r${repeat}.log"
    local matrix_file="$RESULT_DIR/matrices/${kind}_${size}_stage${stage}_r${repeat}.dat"

    echo "Running ${kind}: size=${size}, stage=${stage}, repeat=${repeat}"

    if [[ "$binary" == "./therm_gpu" ]]; then
        "$binary" --size "$size" --eps "$EPS" --iterations "$ITERATIONS" \
            --stage "$stage" --check-interval "$CHECK_INTERVAL" \
            --output "$matrix_file" > "$log_file"
    else
        "$binary" --size "$size" --eps "$EPS" --iterations "$ITERATIONS" \
            --output "$matrix_file" > "$log_file"
    fi

    local time_sec
    local error
    local iterations
    local comment
    time_sec="$(extract_value "Time" "$log_file")"
    error="$(extract_value "Error" "$log_file")"
    iterations="$(extract_value "Iterations" "$log_file")"
    comment="$(extract_comment "$log_file")"
    comment="${comment:-baseline}"

    echo "${kind},${size},${stage},${repeat},${time_sec},${error},${iterations},\"${comment}\"" >> "$CSV"

    if [[ "$kind" == "gpu_stage" ]]; then
        echo "${stage},${size},${repeat},${time_sec},${error},${iterations},\"${comment}\"" >> "$GPU_STAGES_CSV"
    fi
}

echo "Building with CXX=${CXX}"
make clean
make CXX="$CXX" all

for repeat in $(seq 1 "$REPEATS"); do
    for size in "${CPU_ONE_SIZES[@]}"; do
        run_case "cpu_onecore" "./therm_host" "$size" 0 "$repeat"
    done

    for size in "${COMMON_SIZES[@]}"; do
        run_case "cpu_multicore" "./therm_multicore" "$size" 0 "$repeat"
    done

    for stage in "${GPU_STAGES[@]}"; do
        run_case "gpu_stage" "./therm_gpu" "$GPU_OPTIMIZATION_SIZE" "$stage" "$repeat"
    done

    for size in "${COMMON_SIZES[@]}"; do
        run_case "gpu_optimized" "./therm_gpu" "$size" 4 "$repeat"
    done
done

if [[ "$PROFILE" == "1" ]]; then
    for stage in "${GPU_STAGES[@]}"; do
        echo "Profiling GPU stage ${stage}"
        nsys profile -o "$RESULT_DIR/profiles/gpu_stage_${stage}_${GPU_OPTIMIZATION_SIZE}" \
            ./therm_gpu --size "$GPU_OPTIMIZATION_SIZE" --eps "$EPS" \
            --iterations "$PROFILE_ITERATIONS" --stage "$stage" \
            --check-interval "$CHECK_INTERVAL" \
            --output "$RESULT_DIR/matrices/profile_gpu_stage_${stage}.dat"
    done
fi

echo "Saved main results to $CSV"
echo "Saved GPU optimization results to $GPU_STAGES_CSV"
