#!/usr/bin/env python3

import csv
import json
import pickle
import sys
import subprocess
import importlib.util
import traceback
from pathlib import Path

import numpy as np


ROOT = Path("/home/mivia/Desktop/UR-Application/Federica/hand_tracking_test")

INPUT = ROOT / "dataset/nut_assembly_osvi_offset21"
OUTPUT = ROOT / "dataset/nut_assembly_osvi_offset21_release_x"

OLD_WORK = ROOT / "nut_assembly_test_offset21"
NEW_WORK = ROOT / "nut_assembly_test_offset21_release_x"

RECON_SCRIPT = ROOT / "utils/reconstruct_right_hand_3d.py"
PLOT_SCRIPT = ROOT / "utils/plot_osvi_pkl_on_front.py"

SUMMARY = NEW_WORK / "shift_summary.csv"

# Risoluzioni
SCREEN_WIDTH = 2800
FRONT_WIDTH = 672

# Differenze orizzontali misurate sullo screenshot
SHIFT_SCREEN_PX = {
    0: 179,
    3: 179,
    6: 179,
    1: 91,
    4: 91,
    7: 91,
}

NEW_WORK.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Caricamento calibrazione originale camera FRONT
# --------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "recon_geometry",
    RECON_SCRIPT
)

recon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recon)

T_BASE_CAMERA = np.asarray(
    recon.T_BASE_CAMERA,
    dtype=np.float64
)

T_CAMERA_BASE = np.linalg.inv(T_BASE_CAMERA)

FX = float(recon.FX)


# --------------------------------------------------
# Spostamento orizzontale nel piano immagine
# --------------------------------------------------

def shift_point_image_x(xyz_base, delta_u):

    xyz_base = np.asarray(xyz_base, dtype=np.float64)

    if xyz_base.shape != (3,) or not np.isfinite(xyz_base).all():
        raise ValueError("Waypoint XYZ non valido")

    # base_link -> camera FRONT
    p_base = np.array([
        xyz_base[0],
        xyz_base[1],
        xyz_base[2],
        1.0
    ])

    p_cam = T_CAMERA_BASE @ p_base

    depth = float(p_cam[2])

    if not np.isfinite(depth) or depth <= 0:
        raise ValueError(f"Depth non valida: {depth}")

    # Proiezione originale (solo U)
    old_u = FX * p_cam[0] / depth + float(recon.CX)

    # Spostamento orizzontale:
    # manteniamo invariata la profondità
    p_cam_new = p_cam.copy()

    p_cam_new[0] += delta_u * depth / FX

    new_u = FX * p_cam_new[0] / depth + float(recon.CX)

    # camera FRONT -> base_link
    p_base_new = T_BASE_CAMERA @ p_cam_new

    xyz_new = p_base_new[:3]

    if not np.isfinite(xyz_new).all():
        raise ValueError("Risultato XYZ non valido")

    return xyz_new, old_u, new_u, depth


# --------------------------------------------------
# Elaborazione PKL
# --------------------------------------------------

results = []

pkls = sorted(INPUT.glob("task_*/traj*.pkl"))

print("=" * 70)
print("NUT ASSEMBLY - RELEASE X CORRECTION")
print("=" * 70)
print(f"PKL trovati: {len(pkls)}")
print(f"Input:  {INPUT}")
print(f"Output: {OUTPUT}")
print("=" * 70, flush=True)

for index, src in enumerate(pkls, 1):

    task = src.parent.name
    traj_name = src.stem

    task_id = int(task.split("_")[-1])

    dst = OUTPUT / task / src.name

    old_work = OLD_WORK / task / traj_name
    new_work = NEW_WORK / task / traj_name

    old_report = old_work / "human_osvi_build_report.json"
    new_report = new_work / "human_osvi_build_report.json"

    plot_path = new_work / "traj_on_front_final.png"

    dst.parent.mkdir(parents=True, exist_ok=True)
    new_work.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[{index}/{len(pkls)}] {task}/{traj_name}",
        flush=True
    )

    row = {
        "task": task,
        "trajectory": traj_name,
        "status": "FAILED",
        "shift_screen_px": SHIFT_SCREEN_PX.get(task_id, 0),
        "shift_front_px": 0.0,
        "grasp_frame": "",
        "peak_frame": "",
        "first_modified_frame": "",
        "modified_points": 0,
        "plot_status": "NOT_RUN",
        "error": ""
    }

    try:
        if not old_report.exists():
            raise FileNotFoundError(old_report)

        with src.open("rb") as f:
            data = pickle.load(f)

        report = json.loads(old_report.read_text())

        traj = data["traj"]
        n = len(traj)

        pick_frame = int(report["pick_frame"])

        row["grasp_frame"] = pick_frame

        if not (0 <= pick_frame < n - 2):
            raise ValueError("Frame di grasp non valido")

        # -----------------------------------------
        # Coordinate XYZ originali
        # -----------------------------------------

        positions = np.array([
            traj.get(t)["obs"]["eef_pos"]
            for t in range(n)
        ], dtype=np.float64)

        if positions.shape != (n, 3):
            raise ValueError("Shape posizioni non valida")

        if not np.isfinite(positions).all():
            raise ValueError("Waypoint non finiti")

        # -----------------------------------------
        # Rilevamento inizio fase discesa
        # -----------------------------------------

        # Cerchiamo il massimo Z dopo il grasp,
        # escludendo l'ultimo frame di rilascio.
        search_start = pick_frame + 1
        search_end = n - 1

        peak_frame = search_start + int(
            np.argmax(positions[search_start:search_end, 2])
        )

        first_modified = peak_frame + 1

        row["peak_frame"] = peak_frame
        row["first_modified_frame"] = first_modified

        # -----------------------------------------
        # Calcolo shift pixel FRONT
        # -----------------------------------------

        delta_screen = SHIFT_SCREEN_PX.get(task_id, 0)

        delta_u = delta_screen * FRONT_WIDTH / SCREEN_WIDTH

        row["shift_front_px"] = round(delta_u, 5)

        modified = 0
        delta_xyz_samples = []

        if delta_screen != 0:

            for t in range(first_modified, n):

                old = traj._data[t]

                obs, reward, done, info, action = old

                old_xyz = positions[t]

                new_xyz, old_u, new_u, depth = (
                    shift_point_image_x(old_xyz, delta_u)
                )

                obs_new = dict(obs)
                obs_new["eef_pos"] = new_xyz.copy()

                action_new = np.asarray(
                    action, dtype=np.float64
                ).copy()

                if action_new.shape != (8,):
                    raise ValueError(
                        f"Frame {t}: action shape {action_new.shape}"
                    )

                action_new[:3] = new_xyz

                traj._data[t] = (
                    obs_new,
                    reward,
                    done,
                    info,
                    action_new
                )

                delta_xyz_samples.append(new_xyz - old_xyz)

                modified += 1

        row["modified_points"] = modified

        # -----------------------------------------
        # Salvataggio nuovo PKL
        # -----------------------------------------

        with dst.open("wb") as f:
            pickle.dump(
                data,
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        # Verifica salvataggio
        with dst.open("rb") as f:
            check = pickle.load(f)

        if len(check["traj"]) != n:
            raise ValueError("Lunghezza PKL modificata")

        for t in range(n):

            saved_step = check["traj"].get(t)

            saved_xyz = np.asarray(
                saved_step["obs"]["eef_pos"]
            )

            saved_action = np.asarray(
                saved_step["action"]
            )

            if not np.allclose(
                saved_xyz,
                saved_action[:3],
                atol=1e-9
            ):
                raise ValueError(
                    f"Incoerenza stato/azione frame {t}"
                )

            if t < first_modified or delta_screen == 0:

                if not np.allclose(
                    saved_xyz,
                    positions[t],
                    atol=1e-9
                ):
                    raise ValueError(
                        f"Waypoint non previsto modificato: {t}"
                    )

        # -----------------------------------------
        # Nuovo report
        # -----------------------------------------

        report["output"] = str(dst)

        report["release_x_correction"] = {
            "enabled": bool(delta_screen != 0),
            "screen_width": SCREEN_WIDTH,
            "front_width": FRONT_WIDTH,
            "shift_screen_px": delta_screen,
            "shift_front_px": delta_u,
            "grasp_frame": pick_frame,
            "peak_frame": peak_frame,
            "first_modified_frame": first_modified,
            "modified_points": modified,
            "method": "horizontal_image_shift_preserve_camera_depth"
        }

        new_report.write_text(
            json.dumps(report, indent=2) + "\n"
        )

        # -----------------------------------------
        # Plot sul primo frame FRONT
        # -----------------------------------------

        plot_cmd = [
            sys.executable,
            str(PLOT_SCRIPT),
            "--pkl", str(dst),
            "--output", str(plot_path),
            "--geometry-script", str(RECON_SCRIPT),
            "--pick-json", str(new_report),
            "--show-index",
            "--state", "osvi"
        ]

        proc = subprocess.run(
            plot_cmd,
            capture_output=True,
            text=True,
            cwd=ROOT
        )

        if proc.returncode == 0 and plot_path.exists():
            row["plot_status"] = "SUCCESS"
        else:
            row["plot_status"] = "FAILED"

            (new_work / "plot_error.log").write_text(
                proc.stdout + "\n" + proc.stderr
            )

        row["status"] = "SUCCESS"

        print(f"  Grasp frame:       {pick_frame}")
        print(f"  Peak Z frame:      {peak_frame}")
        print(f"  First shift frame: {first_modified}")
        print(f"  Offset FRONT:      +{delta_u:.3f} px")
        print(f"  Punti modificati:  {modified}")

        if delta_xyz_samples:
            delta = delta_xyz_samples[0]

            print(
                f"  Delta base_link: "
                f"X={delta[0]:+.5f} "
                f"Y={delta[1]:+.5f} "
                f"Z={delta[2]:+.5f} m"
            )

        print(f"  Plot: {row['plot_status']}", flush=True)

    except Exception as e:

        row["error"] = str(e)

        print(f"  [FAILED] {e}", flush=True)

        (new_work / "error.log").write_text(
            traceback.format_exc()
        )

    results.append(row)

    with SUMMARY.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys())
        )

        writer.writeheader()
        writer.writerows(results)


# --------------------------------------------------
# Riepilogo finale
# --------------------------------------------------

successes = sum(
    r["status"] == "SUCCESS"
    for r in results
)

plots = sum(
    r["plot_status"] == "SUCCESS"
    for r in results
)

print("\n" + "=" * 70)
print("RISULTATI")
print("=" * 70)

print(f"PKL elaborati: {len(results)}")
print(f"SUCCESS: {successes}")
print(f"FAILED: {len(results) - successes}")
print(f"PLOT: {plots}")

print("\nPKL:", OUTPUT)
print("DEBUG:", NEW_WORK)
print("CSV:", SUMMARY)
