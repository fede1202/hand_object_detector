#!/usr/bin/env python3
"""Plot saved base_link eef_pos trajectory back onto first frontal PKL RGB frame.

Uses the SAME intrinsics and T_BASE_CAMERA as reconstruct_right_hand_3d.py.
Only unpickle trusted files. Does not change input PKL.
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_geometry(path):
    spec = importlib.util.spec_from_file_location('human_3d_geometry', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_rgb(value):
    array = np.asarray(value)

    if array.ndim == 3 and array.shape[2] == 3:
        # La camera FRONT nel PKL è BGR.
        # Matplotlib richiede RGB.
        bgr = array.astype(np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raise ValueError(f'Unexpected frontal image dimensions: {array.shape}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkl', type=Path, required=True, help='Generated human_osvi PKL')
    parser.add_argument('--output', type=Path, required=True, help='Output PNG file')
    parser.add_argument('--geometry-script', type=Path, required=True,
                        help='Existing utils/reconstruct_right_hand_3d.py, providing camera calibration')
    parser.add_argument('--pick-json', type=Path, default=None, help='Optional pick_matched_result.json')
    parser.add_argument('--show-index', action='store_true')
    parser.add_argument('--state', choices=['osvi', 'action'], default='osvi',
                        help='osvi: display gripper state read by loader from action[t+1]; action: raw action[t]')
    parser.add_argument(
        '--pick-shift-xyz',
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=('DX', 'DY', 'DZ'),
        help='Traslazione del solo waypoint di pick, applicata esclusivamente al plot'
    )
    args = parser.parse_args()

    with args.pkl.open('rb') as stream:
        obj = pickle.load(stream)
    traj = obj['traj']
    n = len(traj)
    if n < 2:
        raise ValueError('Need at least two frames')
    first = traj.get(0)['obs']
    rgb = decode_rgb(first['camera_front_image'])
    h, w = rgb.shape[:2]
    geo = load_geometry(args.geometry_script)
    if (h, w) != (geo.H, geo.W):
        raise ValueError(f'Frame size {(w,h)} differs from calibration {(geo.W,geo.H)}')

    positions = np.stack([np.asarray(traj.get(t)['obs']['eef_pos'], dtype=float) for t in range(n)])
    actions = np.stack([np.asarray(traj.get(t)['action'], dtype=float) for t in range(n)])
    if positions.shape != (n, 3) or actions.shape != (n, 8):
        raise ValueError(f'Invalid shapes: positions {positions.shape}, actions {actions.shape}')
        pick = None
    if args.pick_json is not None:
        pick_obj = json.loads(args.pick_json.read_text())
        pick = pick_obj.get('pick_frame')
        if pick is not None:
            pick = int(pick)
            if not 0 <= pick < n:
                raise ValueError(f'Invalid pick frame {pick} for {n} frames')
    # Modifica soltanto le coordinate utilizzate per il plot.
    # Il PKL originale non viene modificato.
    positions = positions.copy()

    if pick is not None:
        shift = np.asarray(args.pick_shift_xyz, dtype=float)

        print("Pick XYZ originale:", positions[pick])
        positions[pick] += shift
        print("Pick XYZ modificato per il plot:", positions[pick])
    # T_BASE_CAMERA maps camera coordinates to base_link. Reprojection needs the inverse.
    t_camera_base = np.linalg.inv(np.asarray(geo.T_BASE_CAMERA, dtype=float))
    pts_base_h = np.column_stack([positions, np.ones(n)])
    pts_cam = (t_camera_base @ pts_base_h.T).T
    z = pts_cam[:, 2]
    finite = np.isfinite(pts_cam).all(axis=1) & (z > 0)
    u = np.full(n, np.nan)
    v = np.full(n, np.nan)
    u[finite] = geo.FX * pts_cam[finite, 0] / z[finite] + geo.CX
    v[finite] = geo.FY * pts_cam[finite, 1] / z[finite] + geo.CY
    visible = finite & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    if args.state == 'osvi':
        # In existing OSVI training loader, grasp[t] is derived from action[t+1][-1].
        states = np.concatenate([actions[1:, -1], [np.nan]])
        state_label = 'Stato OSVI (action[t+1])'
    else:
        states = actions[:, -1]
        state_label = 'Comando action[t]'
    closed = np.isfinite(states) & (states > 0.01)
    open_ = np.isfinite(states) & ~closed

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(rgb, origin='upper')
    # Draw segments only when consecutive endpoints are visible, avoiding image-spanning artifacts.
    for i in range(n - 1):
        if visible[i] and visible[i + 1]:
            ax.plot([u[i], u[i+1]], [v[i], v[i+1]], color='white', linewidth=1.4, alpha=.85, zorder=2)
    for mask, marker, color, label in (
        (visible & open_, 'o', 'tab:cyan', 'Aperto'),
        (visible & closed, 'x', 'tab:orange', 'Chiuso'),
        (visible & ~np.isfinite(states), '.', 'tab:gray', 'Stato non definito'),
    ):
        if mask.any():
            ax.scatter(u[mask], v[mask], s=55, marker=marker, color=color,
                       edgecolors='black' if marker == 'o' else None,
                       linewidths=0.7, label=label, zorder=3)
    if visible[0]:
        ax.scatter(u[0], v[0], marker='s', color='lime', edgecolors='black', s=140, label='Start', zorder=5)
    if visible[-1]:
        ax.scatter(u[-1], v[-1], marker='D', color='magenta', edgecolors='black', s=125, label='End', zorder=5)
    if pick is not None and visible[pick]:
        ax.scatter(u[pick], v[pick], marker='*', s=300, color='yellow', edgecolors='black',
                   label=f'Pick frame {pick}', zorder=6)
        ax.annotate(f'PICK {pick}', (u[pick], v[pick]), xytext=(8, -14),
                    textcoords='offset points', color='yellow', fontsize=11, weight='bold')
    if args.show_index:
        for i in np.flatnonzero(visible):
            ax.text(u[i]+4, v[i]+4, str(i), fontsize=7, color='white', weight='bold',
                    bbox={'facecolor': 'black', 'alpha': .55, 'pad': 1}, zorder=7)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel('u [pixel]')
    ax.set_ylabel('v [pixel]')
    ax.set_title(f'Traiettoria eef_pos riproiettata su front frame 0 — {state_label}')
    ax.legend(loc='upper right', facecolor='white', framealpha=.9)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print('PKL:', args.pkl)
    print('Frames:', n, '| projected on image:', int(visible.sum()), '| outside/behind/invalid:', int((~visible).sum()))
    print('Raw gripper transitions:', [(i, float(actions[i, -1])) for i in range(n)
                                     if i == 0 or actions[i,-1] != actions[i-1,-1]])
    if pick is not None:
        print('Pick frame:', pick, '| visible:', bool(visible[pick]), '| pixel:', (u[pick], v[pick]))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
