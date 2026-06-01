#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_input_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    script_relative = SCRIPT_DIR / path
    if script_relative.exists():
        return script_relative
    return path


def resolve_output_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def read_rows(path):
    path = resolve_input_path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def average_times(rows):
    grouped = defaultdict(list)
    meta = {}

    for row in rows:
        key = (row["kind"], int(row["size"]), int(row["stage"]))
        grouped[key].append(float(row["time_sec"]))
        meta[key] = row

    result = {}
    for key, values in grouped.items():
        result[key] = {
            "time_sec": mean(values),
            "error": float(meta[key]["error"]),
            "iterations": int(meta[key]["iterations"]),
            "comment": meta[key]["comment"],
        }
    return result


def read_gpu_stage_profiles(profile_dir, size, max_iterations=None):
    profile_dir = resolve_input_path(profile_dir)
    if not profile_dir.exists():
        raise FileNotFoundError(f"GPU stage profile directory not found: {profile_dir}")

    if max_iterations is None:
        pattern = f"gpu_stage_profiles_{size}_*.csv"
    else:
        pattern = f"gpu_stage_profiles_{size}_{max_iterations}.csv"

    candidates = sorted(profile_dir.glob(pattern))
    if not candidates:
        fallback_pattern = f"gpu_stage_profiles_{size}_*.csv"
        fallback_candidates = sorted(profile_dir.glob(fallback_pattern))
        if not fallback_candidates:
            raise FileNotFoundError(
                f"No GPU stage profile CSV found by pattern: {profile_dir / pattern}"
            )
        candidates = fallback_candidates

    csv_path = candidates[-1]
    rows = read_rows(csv_path)
    result = {}
    for row in rows:
        if not row.get("stage"):
            continue
        stage = int(row["stage"])
        result[stage] = {
            "time_sec": float(row["time_sec"]),
            "error": float(row["error"]),
            "iterations": int(row["iterations"]),
            "max_iterations": int(row["max_iterations"]),
            "comment": row["comment"],
            "profile": row.get("profile", ""),
            "source": csv_path,
        }
    if not result:
        raise ValueError(f"GPU stage profile CSV has no stage rows: {csv_path}")
    return result


def values_for(data, kind, sizes, stage=0):
    values = []
    for size in sizes:
        item = data.get((kind, size, stage))
        values.append(item["time_sec"] if item else None)
    return values


def gpu_profile_stage_values(stage_profiles):
    stages = sorted(stage_profiles)
    values = [stage_profiles[stage]["time_sec"] for stage in stages]
    return stages, values


def draw_grouped_bars(ax, labels, series):
    width = 0.8 / len(series)
    x_positions = list(range(len(labels)))

    for index, (name, values) in enumerate(series):
        shift = (index - (len(series) - 1) / 2) * width
        xs = [x + shift for x in x_positions]
        heights = [0.0 if value is None else value for value in values]
        bars = ax.bar(xs, heights, width, label=name)

        for bar, value in zip(bars, values):
            if value is None:
                bar.set_alpha(0.15)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Время, c")
    ax.grid(axis="y", alpha=0.35)
    ax.legend()


def annotate_bars(ax):
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height <= 0:
                continue
            ax.annotate(
                f"{height:.3g}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def save_cpu_chart(data, sizes, output_dir):
    labels = [f"{size}*{size}" for size in sizes]
    fig, ax = plt.subplots(figsize=(11, 6))
    draw_grouped_bars(
        ax,
        labels,
        [
            ("CPU-onecore", values_for(data, "cpu_onecore", sizes)),
            ("CPU-multicore", values_for(data, "cpu_multicore", sizes)),
        ],
    )
    ax.set_title("Сравнение времени работы CPU-one и CPU-multi")
    fig.tight_layout()
    fig.savefig(output_dir / "cpu_one_vs_multicore.png", dpi=180)
    plt.close(fig)


def save_gpu_stage_chart(stage_profiles, size, output_dir):
    if not stage_profiles:
        raise ValueError("No GPU stage profile data for optimization chart")

    stages, values = gpu_profile_stage_values(stage_profiles)
    labels = [str(stage) for stage in stages]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, label="GPU-time")
    source = next(iter(stage_profiles.values())).get("source")
    title = f"Этапы оптимизации GPU ({size}*{size})"
    if source:
        title += f"\nИсточник: {source}"
    ax.set_title(title)
    ax.set_xlabel("Номер этапа")
    ax.set_ylabel("Время, c")
    ax.grid(axis="y", alpha=0.35)
    annotate_bars(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gpu_optimization_stages.png", dpi=180)
    plt.close(fig)


def save_cpu_gpu_chart(data, sizes, output_dir):
    labels = [f"{size}*{size}" for size in sizes]
    fig, ax = plt.subplots(figsize=(11, 6))
    draw_grouped_bars(
        ax,
        labels,
        [
            ("CPU-onecore", values_for(data, "cpu_onecore", sizes)),
            ("CPU-multicore", values_for(data, "cpu_multicore", sizes)),
            ("GPU", values_for(data, "gpu_optimized", sizes, stage=4)),
        ],
    )
    ax.set_title("Сравнение CPU-one, CPU-multi и GPU")
    ax.set_yscale("log")
    ax.set_ylabel("Время, c (логарифмическая шкала)")
    annotate_bars(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "cpu_multi_gpu_comparison.png", dpi=180)
    plt.close(fig)


def write_report_tables(data, stage_profiles, sizes, gpu_stage_size, output_dir):
    table_path = output_dir / "tables.md"
    with table_path.open("w", encoding="utf-8") as handle:
        handle.write("# Tables for report\n\n")

        for kind, title in [
            ("cpu_onecore", "CPU-onecore"),
            ("cpu_multicore", "CPU-multicore"),
            ("gpu_optimized", "GPU optimized"),
        ]:
            handle.write(f"## {title}\n\n")
            handle.write("| Размер сетки | Время, c | Точность | Количество итераций |\n")
            handle.write("|---|---:|---:|---:|\n")
            stage = 4 if kind == "gpu_optimized" else 0
            for size in sizes:
                item = data.get((kind, size, stage))
                if not item:
                    continue
                handle.write(
                    f"| {size}*{size} | {item['time_sec']:.6f} | "
                    f"{item['error']:.12g} | {item['iterations']} |\n"
                )
            handle.write("\n")

        handle.write("## GPU optimization stages\n\n")
        handle.write("| Этап | Время, c | Точность | Максимальное количество итераций | Комментарий |\n")
        handle.write("|---:|---:|---:|---:|---|\n")
        if not stage_profiles:
            handle.write("| - | - | - | - | Нет данных |\n")
            return
        stages, _ = gpu_profile_stage_values(stage_profiles)
        for stage in stages:
            item = stage_profiles[stage]
            handle.write(
                f"| {stage} | {item['time_sec']:.6f} | {item['error']:.12g} | "
                f"{item['max_iterations']} | {item['comment']} |\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Draw benchmark charts for Lab 6")
    parser.add_argument("--csv", default="results/benchmarks.csv", help="benchmark CSV file")
    parser.add_argument("--out", default="results/plots", help="output directory")
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--gpu-stage-size", type=int, default=512)
    parser.add_argument("--gpu-stage-profiles", default="results/gpu_stage_profiles")
    parser.add_argument("--gpu-stage-iterations", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = resolve_input_path(args.csv)
    output_dir = resolve_output_path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = average_times(read_rows(csv_path))
    stage_profiles = read_gpu_stage_profiles(
        args.gpu_stage_profiles,
        args.gpu_stage_size,
        args.gpu_stage_iterations,
    )

    save_cpu_chart(data, args.sizes, output_dir)
    save_gpu_stage_chart(stage_profiles, args.gpu_stage_size, output_dir)
    save_cpu_gpu_chart(data, args.sizes, output_dir)
    write_report_tables(data, stage_profiles, args.sizes, args.gpu_stage_size, output_dir)

    print(f"Saved charts and tables to {output_dir}")


if __name__ == "__main__":
    main()
