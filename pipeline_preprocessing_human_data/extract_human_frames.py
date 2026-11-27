import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np


CAMERA_KEYS = {
    "front": [
        "camera_front_image",
        "front_camera_image",
        "front_rgb",
    ],
    "left": [
        "left_camera_image",
        "camera_left_image",
    ],
    "right": [
        "right_camera_image",
        "camera_right_image",
    ],
}


def extract_frame(obs, camera):
    for key in CAMERA_KEYS[camera]:
        if key in obs:
            return obs[key], key
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=["front", "left", "right"],
        default=["front"],
    )
    parser.add_argument(
        "--color",
        choices=["rgb", "bgr"],
        default="rgb",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    traj = data["traj"]
    print("Task:", data.get("task_id"))
    print("Step:", len(traj))

    counts = {camera: 0 for camera in args.cameras}

    for i in range(len(traj)):
        step = traj.get(i)
        obs = step["obs"]

        for camera in args.cameras:
            frame, key = extract_frame(obs, camera)

            if frame is None:
                if i == 0:
                    print(
                        f"[WARNING] Camera {camera} non trovata. "
                        f"Chiavi disponibili: {list(obs.keys())}"
                    )
                continue

            frame = np.asarray(frame)

            if frame.ndim != 3 or frame.shape[2] not in (3, 4):
                raise ValueError(
                    f"Frame {i}, camera {camera}: "
                    f"shape non valida {frame.shape}"
                )

            if frame.dtype != np.uint8:
                if np.issubdtype(frame.dtype, np.floating):
                    if frame.max() <= 1.0:
                        frame = frame * 255
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            if args.color == "rgb":
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Front: direttamente in output per demo.py
            # Left/Right: in sottocartelle dedicate
            camera_dir = (
                output_dir if camera == "front"
                else output_dir / camera
            )
            camera_dir.mkdir(parents=True, exist_ok=True)

            filename = camera_dir / f"frame_{i:05d}.jpg"

            if not cv2.imwrite(str(filename), frame):
                raise RuntimeError(f"Errore salvataggio: {filename}")

            if counts[camera] == 0:
                print(
                    f"Camera {camera}: chiave '{key}', "
                    f"shape {frame.shape}"
                )

            counts[camera] += 1

    print("\n=== RISULTATI ===")
    for camera, count in counts.items():
        print(f"{camera}: {count} frame salvati")

    print("Directory:", output_dir)


if __name__ == "__main__":
    main()
