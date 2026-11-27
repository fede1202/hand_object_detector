#!/usr/bin/env python3
"""Estrae da una traiettoria PKL le immagini laterali per il detector originale."""
import argparse
import pickle
from pathlib import Path
import cv2
import numpy as np

KEYS = {'left': 'camera_lateral_left_image', 'right': 'camera_lateral_right_image'}

def decode(raw, encoding):
    a = np.asarray(raw)
    if a.ndim == 3 and a.shape[-1] == 3:
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if encoding == 'rgb' else a.copy()
    buf = np.asarray(raw, dtype=np.uint8).reshape(-1)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Impossibile decodificare immagine compressa')
    return image  # imdecode restituisce sempre BGR

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--encoding', choices=('rgb', 'bgr'), default='rgb')
    args = p.parse_args()
    dest = Path(args.output)
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    traj = data['traj']
    for side, key in KEYS.items():
        folder = dest / side
        folder.mkdir(parents=True, exist_ok=True)
        for i, step in enumerate(traj):
            obs = step['obs']
            if key not in obs:
                raise KeyError(f'Manca {key} al timestep {i}')
            frame = decode(obs[key], args.encoding)
            out = folder / f'frame_{i:05d}.png'
            if not cv2.imwrite(str(out), frame):
                raise OSError(f'Salvataggio fallito: {out}')
        print(f'[{side}] {len(traj)} frame in {folder}')

if __name__ == '__main__':
    main()
