import argparse
import os
import time
import threading
from queue import Queue

import cv2
import torch
from ultralytics import YOLO


def process_frame(model, frame, imgsz=640, conf=0.25):
    with torch.inference_mode():
        results = model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            device="cpu",
            verbose=False
        )
    return results[0].plot()


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    frames = []
    idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append((idx, frame))
        idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("Видео прочиталось пустым. Проверь input.mp4")

    print(f"Read frames: {len(frames)}")
    print(f"FPS: {fps}")

    return frames, fps


def save_video(output_path, frames, fps):
    if len(frames) == 0:
        raise RuntimeError("Нет кадров для сохранения")

    first_frame = frames[0]

    if first_frame is None:
        raise RuntimeError("Первый кадр пустой, видео сохранить нельзя")

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать видео: {output_path}")

    for i, frame in enumerate(frames):
        if frame is None:
            print(f"Warning: frame {i} is None, skipped")
            continue

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)

    writer.release()
    print(f"Saved video: {output_path}")


def run_single(video_path, output_path, model_path):
    start = time.perf_counter()

    frames, fps = read_video(video_path)

    print("Loading model...")
    model = YOLO(model_path)
    print("Model loaded")

    result_frames = []

    total = len(frames)

    for idx, frame in frames:
        print(f"Single: frame {idx + 1}/{total}")
        annotated = process_frame(model, frame)
        result_frames.append(annotated)

    save_video(output_path, result_frames, fps)

    elapsed = time.perf_counter() - start
    print(f"Single-thread time: {elapsed:.3f} sec")

    return elapsed


def worker(input_buffer, output_buffer, model_path, worker_id):
    try:
        print(f"Worker {worker_id}: loading model...")
        model = YOLO(model_path)
        print(f"Worker {worker_id}: model loaded")

        while True:
            item = input_buffer.get()

            if item is None:
                break

            idx, frame = item
            annotated = process_frame(model, frame)

            output_buffer.put(("ok", idx, annotated))

    except Exception as e:
        output_buffer.put(("error", worker_id, str(e)))


def run_multi(video_path, output_path, model_path, workers_count):
    start = time.perf_counter()

    frames, fps = read_video(video_path)

    input_buffer = Queue()
    output_buffer = Queue()

    for item in frames:
        input_buffer.put(item)

    for _ in range(workers_count):
        input_buffer.put(None)

    threads = []

    for worker_id in range(workers_count):
        t = threading.Thread(
            target=worker,
            args=(input_buffer, output_buffer, model_path, worker_id)
        )
        t.start()
        threads.append(t)

    results = {}
    total = len(frames)

    while len(results) < total:
        status, a, b = output_buffer.get()

        if status == "error":
            raise RuntimeError(f"Worker {a} crashed: {b}")

        idx = a
        annotated = b
        results[idx] = annotated

        print(f"Multi: done frame {len(results)}/{total}")

    for t in threads:
        t.join()

    ordered_frames = []

    for idx in range(total):
        ordered_frames.append(results[idx])

    save_video(output_path, ordered_frames, fps)

    elapsed = time.perf_counter() - start
    print(f"Multi-thread time: {elapsed:.3f} sec")
    print(f"Workers: {workers_count}")

    return elapsed


def run_benchmark(video_path, output_path, model_path, max_workers):
    results = []

    for workers in range(1, max_workers + 1):
        out_name = output_path.replace(".mp4", f"_w{workers}.mp4")

        print(f"\nTesting workers = {workers}")

        elapsed = run_multi(
            video_path=video_path,
            output_path=out_name,
            model_path=model_path,
            workers_count=workers
        )

        results.append((workers, elapsed))

    print("\nBenchmark result:")
    print("workers | time | speedup")

    base_time = results[0][1]
    best_workers = None
    best_time = 10**9

    for workers, elapsed in results:
        speedup = base_time / elapsed

        print(f"{workers:7d} | {elapsed:6.3f} | {speedup:6.3f}x")

        if elapsed < best_time:
            best_time = elapsed
            best_workers = workers

    print(f"\nBest workers count: {best_workers}")
    print(f"Best time: {best_time:.3f} sec")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", default="input.mp4", help="Путь к входному видео")
    parser.add_argument("--mode", required=True, choices=["single", "multi", "bench"])
    parser.add_argument("--output", required=True, help="Путь к выходному видео")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    parser.add_argument("--model", default="yolov8s-pose.pt")

    args = parser.parse_args()

    torch.set_num_threads(1)

    if args.mode == "single":
        run_single(args.video, args.output, args.model)

    elif args.mode == "multi":
        run_multi(args.video, args.output, args.model, args.workers)

    elif args.mode == "bench":
        run_benchmark(args.video, args.output, args.model, args.max_workers)


if __name__ == "__main__":
    main()