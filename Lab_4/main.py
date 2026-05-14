import argparse
import logging
import os
import threading
import time
from queue import Queue, Empty

import cv2


os.makedirs("log", exist_ok=True) 
logging.basicConfig(
    filename=os.path.join("log", "app.log"), 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    def __init__(self, cam_name, resolution):
        self._cap = None
        try:
            try:
                src = int(cam_name)
            except (TypeError, ValueError):
                src = cam_name
            self._cap = cv2.VideoCapture(src)
            if not self._cap.isOpened():
                raise RuntimeError(f"Camera '{cam_name}' not found in system")
            width, height = resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logging.info("Camera '%s' opened at %dx%d", cam_name, width, height)
        except Exception as e:
            logging.error("Camera init error: %s", e)
            self._release()
            raise

    def get(self):
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            logging.error("Camera read failed (disconnected?)")
            return None
        return frame

    def _release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logging.info("Camera released")

    def __del__(self):
        self._release()


class WindowImage:
    def __init__(self, fps: int):
        self._fps = max(1, int(fps))
        self._period = 1.0 / self._fps
        self._name = "camera and data"
        try:
            cv2.namedWindow(self._name, cv2.WINDOW_AUTOSIZE)
            logging.info("Window opened (fps=%d)", self._fps)
        except Exception as e:
            logging.error("Window init error: %s", e)
            raise

    @property
    def period(self) -> float:
        return self._period

    def show(self, img) -> None:
        if img is None:
            return
        try:
            cv2.imshow(self._name, img)
        except Exception as e:
            logging.error("Window show error: %s", e)
            raise

    def __del__(self):
        try:
            cv2.destroyWindow(self._name)
            logging.info("Window closed")
        except Exception:
            pass


def sensor_worker(sensor: Sensor, q: Queue, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        value = sensor.get()
        ts = time.perf_counter()
        try:
            if q.full():
                q.get_nowait()
        except Empty:
            pass
        q.put((value, ts))


def camera_worker(cam: SensorCam, q: Queue,
                  stop_event: threading.Event, err_event: threading.Event) -> None:
    while not stop_event.is_set():
        frame = cam.get()
        if frame is None:
            err_event.set()
            break
        ts = time.perf_counter()
        try:
            if q.full():
                q.get_nowait()
        except Empty:
            pass
        q.put((frame, ts))


def parse_args():
    parser = argparse.ArgumentParser(description="Lab4: sensors + camera with threads")
    parser.add_argument("--camName", type=str, default="0",
                        help="Camera in system: index ('0') or path ('/dev/video0')")
    parser.add_argument("--size", nargs=2, type=int, default=[1280, 720],
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Window display frequency")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.info("Starting lab4 (camName=%s, size=%s, fps=%d)",
                 args.camName, args.size, args.fps)

    try:
        cam = SensorCam(args.camName, tuple(args.size))
    except Exception:
        print("[ERROR] Failed to init camera. See log/app.log")
        return

    try:
        window = WindowImage(args.fps)
    except Exception:
        print("[ERROR] Failed to init window. See log/app.log")
        return

    sensors = [SensorX(0.01), SensorX(0.1), SensorX(1)]
    sensor_queues = [Queue(maxsize=1) for _ in sensors]
    cam_queue = Queue(maxsize=1)

    stop_event = threading.Event()
    err_event = threading.Event()

    threads = [
        threading.Thread(target=sensor_worker,
                         args=(s, q, stop_event), daemon=True)
        for s, q in zip(sensors, sensor_queues)
    ]
    threads.append(threading.Thread(
        target=camera_worker,
        args=(cam, cam_queue, stop_event, err_event), daemon=True,
    ))

    for t in threads:
        t.start()

    last_vals = [0, 0, 0]
    last_sensor_ts = [0.0, 0.0, 0.0]
    last_frame = None
    last_frame_ts = 0.0

    avg_lat = [0.0, 0.0, 0.0, 0.0]
    alpha = 0.1

    h = args.size[1]
    try:
        while True:
            loop_start = time.perf_counter()

            if err_event.is_set():
                print("[ERROR] Camera failed during work. See log/app.log")
                break

            for i, q in enumerate(sensor_queues):
                try:
                    val, ts = q.get_nowait()
                    last_vals[i] = val
                    last_sensor_ts[i] = ts
                except Empty:
                    pass

            try:
                frame, ts = cam_queue.get_nowait()
                last_frame = frame
                last_frame_ts = ts
            except Empty:
                pass

            if last_frame is not None:
                now = time.perf_counter()
                lat_s = [(now - t) * 1000 if t > 0 else 0.0 for t in last_sensor_ts]
                lat_f = (now - last_frame_ts) * 1000 if last_frame_ts > 0 else 0.0
                for i in range(3):
                    avg_lat[i] = avg_lat[i] * (1 - alpha) + lat_s[i] * alpha
                avg_lat[3] = avg_lat[3] * (1 - alpha) + lat_f * alpha

                img = last_frame.copy()
                text1 = (f"Sensor0: {last_vals[0]}  "
                         f"Sensor1: {last_vals[1]}  "
                         f"Sensor2: {last_vals[2]}")
                text2 = (f"Lat ms  s0:{avg_lat[0]:.1f} s1:{avg_lat[1]:.1f} "
                         f"s2:{avg_lat[2]:.1f} cam:{avg_lat[3]:.1f}")
                cv2.putText(img, text1, (10, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, text2, (10, h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                window.show(img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.perf_counter() - loop_start
            rest = window.period - elapsed
            if rest > 0:
                time.sleep(rest)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")
    finally:
        print("[INFO] Stopping...")
        logging.info("Stopping threads")
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        logging.info("Done")
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
