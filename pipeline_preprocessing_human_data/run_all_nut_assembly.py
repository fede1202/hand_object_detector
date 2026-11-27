#!/usr/bin/env python3

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/mivia/Desktop/UR-Application/Federica/hand_tracking_test")

# Tutti i PKL umani originali
INPUT = Path(os.environ.get(
    "NUT_INPUT",
    str(ROOT / "dataset/nut_assembly")
))

OUTPUT = ROOT / "dataset/nut_assembly_osvi_full_release_x"
WORK = ROOT / "nut_assembly_full_release_x_debug"

PIPELINE = ROOT / "utils/process_human_osvi_front_grasp_nut_release.py"
PLOT = ROOT / "utils/plot_osvi_pkl_on_front.py"
RECON = ROOT / "utils/reconstruct_right_hand_3d.py"
SUMMARY = WORK / "evaluation_summary.csv"

WORK.mkdir(parents=True, exist_ok=True)

pkls = sorted(INPUT.glob("task_*/traj*.pkl"))

if not pkls:
    raise SystemExit(f"Nessun PKL trovato in {INPUT}")

print(f"[INPUT] {INPUT}")
print(f"[PKL] {len(pkls)}")
print(f"[OUTPUT] {OUTPUT}")
print(f"[DEBUG] {WORK}", flush=True)

results = []

for i, pkl in enumerate(pkls, 1):

    task = pkl.parent.name
    traj = pkl.stem

    output = OUTPUT / task / pkl.name
    work = WORK / task / traj

    report = work / "human_osvi_build_report.json"
    log = work / "pipeline.log"
    plot = work / "traj_on_front_final.png"

    output.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    detection_files = [
        work / camera / filename
        for camera in ("left", "front")
        for filename in (
            "hand_detections.csv",
            "object_detections.csv"
        )
    ]

    # Riutilizza soltanto le detection già presenti
    # in questa nuova cartella di elaborazione.
    skip_detection = all(f.is_file() for f in detection_files)

    print(
        f"[{i}/{len(pkls)}] {task}/{traj} "
        f"reuse_detection={skip_detection}",
        flush=True
    )

    # Evita falsi successi da output precedenti.
    output.unlink(missing_ok=True)
    report.unlink(missing_ok=True)
    plot.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        str(PIPELINE),
        "--pkl", str(pkl),
        "--output-pkl", str(output),
        "--work-dir", str(work),
        "--repo", str(ROOT / "hand_object_detector"),
        "--utils", str(ROOT / "utils"),
        "--open-value", "0.0",
        "--closed-value", "1.0",
        "--hand-thresh", "0.5",
        "--object-thresh", "0.5",
        "--pick-delay", "2",
        "--max-match-distance-px", "60",
        "--front-object-max-distance-px", "60",
        "--grasp-offset-u-px", "0",
        "--grasp-offset-v-px", "-21",
        "--grasp-offset-xyz",
        "0.0", "0.0", "0.0",
        "--depth-patch", "3",
        "--hand-fallback-offset-v-px", "37",
        "--release-shift",
        "--overwrite",
    ]

    if skip_detection:
        cmd.append("--skip-detection")

    with log.open("w") as f:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT
        )

    row = {
        "task": task,
        "trajectory": traj,
        "status": "FAILED",
        "detected_pick": "",
        "grasp_frame": "",
        "grasp_source": "",
        "release_shift_px": "",
        "release_first_frame": "",
        "release_modified_points": "",
        "plot_status": "NOT_RUN",
        "error": "",
    }

    if proc.returncode == 0 and output.is_file() and report.is_file():

        try:
            info = json.loads(report.read_text())
            release = info.get("release_shift", {})

            row["status"] = "SUCCESS"
            row["detected_pick"] = info.get(
                "detected_pick_frame", ""
            )
            row["grasp_frame"] = info.get("pick_frame", "")
            row["grasp_source"] = info.get(
                "grasp_position_source", ""
            )
            row["release_shift_px"] = release.get(
                "shift_front_px", ""
            )
            row["release_first_frame"] = release.get(
                "first_modified_frame", ""
            )
            row["release_modified_points"] = release.get(
                "modified_points", ""
            )

            plot_cmd = [
                sys.executable,
                str(PLOT),
                "--pkl", str(output),
                "--output", str(plot),
                "--geometry-script", str(RECON),
                "--pick-json", str(report),
                "--show-index",
                "--state", "osvi"
            ]

            plotted = subprocess.run(
                plot_cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            row["plot_status"] = (
                "SUCCESS"
                if plotted.returncode == 0 and plot.is_file()
                else "FAILED"
            )

            if row["plot_status"] == "FAILED":
                (work / "plot_error.log").write_text(plotted.stdout)

        except Exception as e:
            row["status"] = "FAILED"
            row["error"] = str(e)

    else:
        lines = log.read_text(errors="replace").splitlines()
        row["error"] = next(
            (line.strip() for line in reversed(lines) if line.strip()),
            "Errore sconosciuto"
        )

    results.append(row)

    # Salva i risultati incrementali
    with SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(results)

    print(
        f"  [{row['status']}] "
        f"pick={row['grasp_frame']} "
        f"shift={row['release_shift_px']} "
        f"plot={row['plot_status']}",
        flush=True
    )

    if row["error"]:
        print("  ERROR:", row["error"][:180], flush=True)

print("\n" + "=" * 60)
print("ELABORAZIONE COMPLETATA")
print("=" * 60)

counts = Counter(r["status"] for r in results)
plots = Counter(r["plot_status"] for r in results)

print("Traiettorie:", len(results))
print("SUCCESS:", counts["SUCCESS"])
print("FAILED:", counts["FAILED"])
print("PLOT:", plots["SUCCESS"])
print("CSV:", SUMMARY)
print("PKL:", OUTPUT)
