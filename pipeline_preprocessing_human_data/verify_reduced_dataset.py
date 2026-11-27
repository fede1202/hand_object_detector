
#!/usr/bin/env python3

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np


CAMERA_KEYS = [
    "camera_front_image",
    "front_camera_image",
    "front_rgb",
]


def compare_values(a, b, path, errors, max_errors=30):
    """Confronto ricorsivo di valori, dizionari e oggetti."""

    if len(errors) >= max_errors:
        return

    if type(a) is not type(b):
        errors.append(
            f"{path}: tipi differenti "
            f"({type(a).__name__} != {type(b).__name__})"
        )
        return

    if isinstance(a, dict):
        keys_a = set(a.keys())
        keys_b = set(b.keys())

        for key in sorted(keys_a - keys_b, key=str):
            errors.append(f"{path}: chiave mancante: {key}")

        for key in sorted(keys_b - keys_a, key=str):
            errors.append(f"{path}: chiave aggiunta: {key}")

        for key in sorted(keys_a & keys_b, key=str):
            compare_values(
                a[key], b[key],
                f"{path}[{key!r}]", errors, max_errors
            )
            if len(errors) >= max_errors:
                return
        return

    if isinstance(a, np.ndarray):
        if a.shape != b.shape:
            errors.append(
                f"{path}: shape {a.shape} != {b.shape}"
            )
        elif a.dtype != b.dtype:
            errors.append(
                f"{path}: dtype {a.dtype} != {b.dtype}"
            )
        elif not np.array_equal(a, b, equal_nan=True):
            errors.append(f"{path}: valori array differenti")
        return

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            errors.append(
                f"{path}: lunghezza {len(a)} != {len(b)}"
            )

        for i in range(min(len(a), len(b))):
            compare_values(
                a[i], b[i], f"{path}[{i}]",
                errors, max_errors
            )
            if len(errors) >= max_errors:
                return
        return

    if hasattr(a, "__dict__"):
        compare_values(
            vars(a), vars(b), path,
            errors, max_errors
        )
        return

    try:
        same = a == b
        if isinstance(same, np.ndarray):
            same = bool(np.all(same))
        if not bool(same):
            errors.append(f"{path}: valori differenti")
    except Exception as exc:
        errors.append(
            f"{path}: confronto non supportato ({exc})"
        )


def get_step(traj, index):
    if hasattr(traj, "get") and not isinstance(traj, dict):
        return traj.get(index)
    return traj[index]


def verify_pkl(original_path, reduced_path):
    """
    Controlla:
    - chiavi e valori del dizionario principale
    - frame mantenuti, senza alterazioni
    - metadati della Trajectory, quando accessibili
    """

    errors = []

    with open(original_path, "rb") as f:
        original = pickle.load(f)

    with open(reduced_path, "rb") as f:
        reduced = pickle.load(f)

    if not isinstance(original, dict) or not isinstance(reduced, dict):
        return False, 0, 0, ["PKL non contenente un dizionario"]

    if "traj" not in original or "traj" not in reduced:
        return False, 0, 0, ["Chiave 'traj' mancante"]

    original_traj = original["traj"]
    reduced_traj = reduced["traj"]

    n_original = len(original_traj)
    n_reduced = len(reduced_traj)

    if n_reduced == 0:
        errors.append("Traiettoria ridotta vuota")

    if n_reduced > n_original:
        errors.append(
            f"Traiettoria aumentata: {n_original} -> {n_reduced}"
        )

    # Controlla tutti gli altri campi del PKL
    original_meta = {
        k: v for k, v in original.items() if k != "traj"
    }
    reduced_meta = {
        k: v for k, v in reduced.items() if k != "traj"
    }

    compare_values(
        original_meta, reduced_meta, "metadata", errors
    )

    if type(original_traj) is not type(reduced_traj):
        errors.append("Tipo dell'oggetto Trajectory modificato")

    # Controlla tutti i frame conservati
    for i in range(min(n_original, n_reduced)):
        if len(errors) >= 30:
            break

        compare_values(
            get_step(original_traj, i),
            get_step(reduced_traj, i),
            f"traj[{i}]",
            errors
        )

    # Controlla gli attributi extra della Trajectory,
    # escludendo solamente il contenitore degli step.
    if (
        type(original_traj) is type(reduced_traj)
        and hasattr(original_traj, "__dict__")
        and hasattr(reduced_traj, "__dict__")
    ):
        orig_attrs = vars(original_traj)
        red_attrs = vars(reduced_traj)

        candidate_storage = [
            "_data", "data", "_traj", "traj",
            "_trajectory", "trajectory", "steps", "_steps"
        ]

        storage = None
        for name in candidate_storage:
            if name in orig_attrs and name in red_attrs:
                try:
                    if (
                        len(orig_attrs[name]) == n_original
                        and len(red_attrs[name]) == n_reduced
                    ):
                        storage = name
                        break
                except (TypeError, ValueError):
                    pass

        if storage is None:
            errors.append(
                "Contenitore interno Trajectory non identificato"
            )
        else:
            extra_orig = {
                k: v for k, v in orig_attrs.items() if k != storage
            }
            extra_red = {
                k: v for k, v in red_attrs.items() if k != storage
            }
            compare_values(
                extra_orig, extra_red,
                "trajectory_attributes", errors
            )

    return len(errors) == 0, n_original, n_reduced, errors


def extract_last_frame(reduced_path, output_dir, color):
    """Estrae l'ultimo frame della camera frontale."""

    with open(reduced_path, "rb") as f:
        data = pickle.load(f)

    traj = data["traj"]

    if len(traj) == 0:
        raise ValueError("Traiettoria vuota")

    last_idx = len(traj) - 1
    step = get_step(traj, last_idx)
    obs = step["obs"]

    frame = None

    for key in CAMERA_KEYS:
        if key in obs:
            frame = np.asarray(obs[key])
            break

    if frame is None:
        raise KeyError("Camera frontale non trovata")

    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"Shape non valida: {frame.shape}")

    if frame.dtype != np.uint8:
        if (
            np.issubdtype(frame.dtype, np.floating)
            and frame.max() <= 1.0
        ):
            frame = frame * 255

        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if color == "rgb":
        conversion = (
            cv2.COLOR_RGB2BGR
            if frame.shape[2] == 3
            else cv2.COLOR_RGBA2BGR
        )
        frame = cv2.cvtColor(frame, conversion)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir /
        f"{reduced_path.stem}_last_frame_{last_idx:05d}.jpg"
    )

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Impossibile salvare {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--reduced", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument(
        "--color",
        choices=["rgb", "bgr"],
        default="bgr"
    )

    args = parser.parse_args()

    original_root = args.original.resolve()
    reduced_root = args.reduced.resolve()
    output_root = args.output.resolve()

    if not original_root.is_dir():
        parser.error(f"Dataset originale assente: {original_root}")

    if not reduced_root.is_dir():
        parser.error(f"Dataset ridotto assente: {reduced_root}")

    if output_root == original_root or output_root == reduced_root:
        parser.error("La cartella dei risultati deve essere separata")

    output_root.mkdir(parents=True, exist_ok=True)

    records = []

    original_files = {
        p.relative_to(original_root): p
        for p in original_root.glob("task_*/*.pkl")
    }

    reduced_files = {
        p.relative_to(reduced_root): p
        for p in reduced_root.glob("task_*/*.pkl")
    }

    valid = 0
    invalid = 0
    missing = 0
    extracted = 0
    image_errors = 0

    print("=" * 65)
    print("VERIFY HUMAN REDUCED DATASET")
    print("=" * 65)
    print(f"Original PKL: {len(original_files)}")
    print(f"Reduced PKL : {len(reduced_files)}")
    print("=" * 65)

    for rel_path, original_path in sorted(original_files.items()):

        reduced_path = reduced_files.get(rel_path)
        task = rel_path.parent.name
        name = rel_path.name

        if reduced_path is None:
            print(f"[MISSING] {task}/{name}")
            missing += 1

            records.append([
                task, name, "MISSING", "", "", "",
                "PKL non presente nel dataset ridotto"
            ])
            continue

        try:
            ok, n_orig, n_red, errors = verify_pkl(
                original_path, reduced_path
            )

            status = "OK" if ok else "ERROR"

            if ok:
                valid += 1
            else:
                invalid += 1

            last_frame = n_red - 1 if n_red else ""

            image_result = ""

            # Estrazione immagini anche se la verifica fallisce:
            # utile per diagnosticare gli errori.
            try:
                image_path = extract_last_frame(
                    reduced_path,
                    output_root / "last_frames" / task,
                    args.color
                )
                image_result = str(image_path)
                extracted += 1

            except Exception as exc:
                image_errors += 1
                errors.append(f"IMAGE ERROR: {exc}")

            print(
                f"[{status}] {task}/{name} | "
                f"{n_orig} -> {n_red} | "
                f"ultimo={last_frame}"
            )

            if errors:
                for error in errors[:5]:
                    print(f"    - {error}")
                if len(errors) > 5:
                    print(f"    ... altre {len(errors) - 5} differenze")

            records.append([
                task, name, status,
                n_orig, n_red, last_frame,
                " | ".join(errors)
            ])

        except Exception as exc:
            invalid += 1
            print(f"[ERROR] {task}/{name}: {exc}")

            records.append([
                task, name, "ERROR", "", "", "", str(exc)
            ])

    # Segnala eventuali PKL presenti solo nel reduced
    extra_files = set(reduced_files) - set(original_files)

    for rel_path in sorted(extra_files):
        print(f"[EXTRA] {rel_path}")

        records.append([
            rel_path.parent.name,
            rel_path.name,
            "EXTRA", "", "", "",
            "File senza corrispondente originale"
        ])

    report_path = output_root / "verification_report.csv"

    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task", "trajectory", "status",
            "original_frames", "reduced_frames",
            "last_frame", "details"
        ])
        writer.writerows(records)

    summary = (
        "\n" + "=" * 65 + "\n"
        "DATASET VERIFICATION SUMMARY\n"
        + "=" * 65 + "\n"
        f"Original PKL       : {len(original_files)}\n"
        f"Reduced PKL        : {len(reduced_files)}\n"
        f"Verified OK        : {valid}\n"
        f"Verification errors: {invalid}\n"
        f"Missing PKL        : {missing}\n"
        f"Extra PKL          : {len(extra_files)}\n"
        f"Images extracted   : {extracted}\n"
        f"Image errors       : {image_errors}\n"
        f"Report CSV         : {report_path}\n"
        + "=" * 65
    )

    print(summary)

    (output_root / "verification_summary.txt").write_text(
        summary + "\n"
    )


if __name__ == "__main__":
    main()
