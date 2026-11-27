#!/usr/bin/env python3
"""Reconstruct right-hand bbox-center 3D points in ROS base_link from aligned ZED RGB-depth.
This is geometric validation ONLY; outputs are NOT robot motion commands.
"""
import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W, H = 672, 376
FX = FY = 363.8398742675781
CX, CY = 337.5829162597656, 178.53465270996094
T_BASE_CAMERA = np.array([
    [-0.9995271385, 0.0017188839, -0.0307008924, 0.0258804482],
    [ 0.0210472614, 0.7661305314, -0.6423402694, 1.0325620478],
    [ 0.0224167827,-0.6426827011, -0.7658044356, 0.4595237268],
    [ 0.0,          0.0,           0.0,          1.0],
], dtype=np.float64)


def parse_index(frame):
    m = re.fullmatch(r'frame_(\d+)\.(?:jpg|jpeg|png)', frame, flags=re.I)
    if not m:
        raise ValueError(f'Nome frame non riconosciuto: {frame!r}')
    return int(m.group(1))


def num(raw):
    try:
        return float(raw)
    except (ValueError, TypeError):
        return np.nan


def median_patch(depth, u, v, radius, min_valid, min_z, max_z):
    if not np.isfinite([u, v]).all() or not (0 <= u < W and 0 <= v < H):
        return np.nan, 0.0, 0, 'invalid_pixel'
    ix, iy = int(round(u)), int(round(v))
    ix, iy = np.clip(ix, 0, W - 1), np.clip(iy, 0, H - 1)
    patch = depth[max(0, iy-radius):min(H, iy+radius+1),
                  max(0, ix-radius):min(W, ix+radius+1)]
    valid = np.isfinite(patch) & (patch >= min_z) & (patch <= max_z)
    count = int(valid.sum())
    frac = count / patch.size
    if frac < min_valid or count == 0:
        return np.nan, frac, count, 'insufficient_depth'
    return float(np.median(patch[valid])), frac, count, 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--csv', required=True, help='right_hand_actions.csv in pixel coordinates')
    ap.add_argument('--output', required=True, help='output directory')
    ap.add_argument('--patch', type=int, default=7, help='odd median filter window')
    ap.add_argument('--min-valid', type=float, default=0.5, help='minimum fraction of valid depth pixels')
    ap.add_argument('--min-depth', type=float, default=0.15)
    ap.add_argument('--max-depth', type=float, default=2.5)
    ap.add_argument('--max-step', type=float, default=0.08, help='flag consecutive-frame displacements above this many meters')
    args = ap.parse_args()
    if args.patch < 1 or args.patch % 2 == 0:
        ap.error('--patch must be a positive odd integer')
    if not 0 <= args.min_valid <= 1:
        ap.error('--min-valid must be between 0 and 1')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.csv, newline='') as f:
        inputs = list(csv.DictReader(f))
    if not inputs:
        raise ValueError('CSV vuoto')
    if not {'frame', 'x', 'y', 'gripper'}.issubset(inputs[0]):
        raise ValueError('Il CSV deve contenere frame,x,y,gripper')

    # Only load trusted self-generated PKL files.
    with open(args.pkl, 'rb') as f:
        obj = pickle.load(f)
    trajectory = obj['traj'] if isinstance(obj, dict) and 'traj' in obj else obj
    n = len(trajectory)
    print(f'PKL: {args.pkl} — {n} step')
    print(f'CSV: {args.csv} — {len(inputs)} frame')
    if len(inputs) != n:
        raise ValueError('Il numero di righe CSV deve coincidere con il numero di step PKL')
    indices = [parse_index(r['frame']) for r in inputs]
    if indices != list(range(n)):
        raise ValueError('I frame CSV non corrispondono agli indici 0..N-1 del PKL')

    fields = ['frame', 'x', 'y', 'z', 'gripper', 'valid', 'u', 'v', 'depth_m',
              'depth_valid_fraction', 'depth_valid_pixels', 'status']
    validation_fields = ['frame', 'valid', 'status', 'depth_m', 'depth_valid_fraction',
                         'step_m', 'jump_over_threshold', 'gripper', 'gripper_transition']
    records, checks = [], []
    prev_xyz = None
    prev_valid = False
    prev_gripper = None

    for i, r in enumerate(inputs):
        step = trajectory.get(i) if hasattr(trajectory, 'get') else trajectory[i]
        obs = step['obs']
        depth = np.asarray(obs['camera_front_depth'])
        if depth.shape != (H, W):
            raise ValueError(f'Frame {i}: depth shape {depth.shape} != {(H, W)}')
        if i == 0 and 'camera_front_image' in obs and np.asarray(obs['camera_front_image']).shape[:2] != (H, W):
            raise ValueError('RGB frontale non ha la stessa dimensione della depth')
        u, v = num(r['x']), num(r['y'])
        g = int(r['gripper'])
        if g not in (0, 1):
            raise ValueError(f'Frame {i}: gripper deve essere 0 o 1')
        z, fraction, count, status = median_patch(depth, u, v, args.patch // 2,
                                                  args.min_valid, args.min_depth, args.max_depth)
        xyz = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        if status == 'ok':
            camera_xyz = np.array([(u - CX)*z/FX, (v - CY)*z/FY, z, 1.0])
            xyz = (T_BASE_CAMERA @ camera_xyz)[:3]
            if not np.isfinite(xyz).all():
                status = 'invalid_projection'
        valid = (status == 'ok')
        step_m = float(np.linalg.norm(xyz - prev_xyz)) if valid and prev_valid else np.nan
        jump = bool(np.isfinite(step_m) and step_m > args.max_step)
        transition = ('start' if prev_gripper is None else
                      'close' if prev_gripper == 0 and g == 1 else
                      'open' if prev_gripper == 1 and g == 0 else 'none')
        records.append(dict(frame=r['frame'], x=xyz[0], y=xyz[1], z=xyz[2],
                            gripper=g, valid=int(valid), u=u, v=v, depth_m=z,
                            depth_valid_fraction=fraction, depth_valid_pixels=count,
                            status=status))
        checks.append(dict(frame=r['frame'], valid=int(valid), status=status,
                           depth_m=z, depth_valid_fraction=fraction, step_m=step_m,
                           jump_over_threshold=int(jump), gripper=g,
                           gripper_transition=transition))
        prev_xyz, prev_valid, prev_gripper = xyz, valid, g

    for filename, columns, rows in [('right_hand_3d.csv', fields, records),
                                    ('right_hand_3d_validation.csv', validation_fields, checks)]:
        with open(out / filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(1, len(records)):
        a, b = records[i-1], records[i]
        if a['valid'] and b['valid']:
            ax.plot([a['x'], b['x']], [a['y'], b['y']], [a['z'], b['z']],
                    color='0.6', linewidth=1, alpha=0.65)
    for g, label, color in [(0, 'Gripper aperto', 'tab:blue'), (1, 'Gripper chiuso', 'tab:orange')]:
        points = [r for r in records if r['valid'] and r['gripper'] == g]
        if points:
            ax.scatter([r['x'] for r in points], [r['y'] for r in points],
                       [r['z'] for r in points], c=color, s=25, label=label, depthshade=False)
    for idx in [0, len(records)-1]:
        r = records[idx]
        if r['valid']:
            ax.text(r['x'], r['y'], r['z'], f'  frame {idx}', fontsize=9)
    ax.set_xlabel('X base_link [m]')
    ax.set_ylabel('Y base_link [m]')
    ax.set_zlabel('Z base_link [m]')
    ax.set_title('Traiettoria mano destra — punti 3D (non TCP robot)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'right_hand_3d_plot.png', dpi=180)
    plt.close(fig)

    print(f'Punti 3D validi: {sum(r["valid"] for r in records)}/{len(records)}')
    print(f'Salti > {args.max_step:.3f} m: {sum(c["jump_over_threshold"] for c in checks)}')
    print('Transizioni gripper:', [(i, c['gripper_transition']) for i, c in enumerate(checks)
                                   if c['gripper_transition'] in ('open', 'close')])
    if not any(c['gripper_transition'] == 'open' for c in checks):
        print('ATTENZIONE: nessuna apertura del gripper rilevata; NON eseguire sul robot.')
    print(f'Output: {out.resolve()}')
    print('NOTA: calibrazione valida solo se relativa al setup di acquisizione; nessuna validazione robotica eseguita.')


if __name__ == '__main__':
    main()
