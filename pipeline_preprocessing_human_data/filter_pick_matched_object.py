#!/usr/bin/env python3
"""Trova il primo frame con mano destra Portable e oggetto vicino in due camere.

La distanza è quella euclidea, in pixel, fra i centri delle bbox
rilevate per la mano e per l'oggetto (NON l'offset predetto dalla rete).
Nessun filtro per dimensioni, area o aspect ratio della bbox.
"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def index(name):
    m = re.search(r'(\d+)$', Path(str(name)).stem)
    if not m:
        raise ValueError(f'frame non valido: {name}')
    return int(m.group(1))


def load(path):
    out = defaultdict(list)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            out[index(row['frame'])].append(row)
    return out


def inspect(t, hands, objects, hand_thresh, obj_thresh, max_distance):
    right = [
        h for h in hands.get(t, [])
        if int(float(h['side'])) == 1
        and int(float(h['contact_state'])) == 3
        and float(h['score']) >= hand_thresh
    ]
    objs = [o for o in objects.get(t, []) if float(o['score']) >= obj_thresh]
    if not right:
        return {'valid': 0, 'reason': 'no_right_portable_hand'}
    if not objs:
        return {'valid': 0, 'reason': 'no_detected_object'}

    candidates = []
    for h in right:
        hx, hy = float(h['cx']), float(h['cy'])
        for o in objs:
            ox, oy = float(o['cx']), float(o['cy'])
            distance = math.hypot(hx - ox, hy - oy)
            if math.isfinite(distance):
                candidates.append((distance, -float(h['score']), -float(o['score']), h, o))
    if not candidates:
        return {'valid': 0, 'reason': 'invalid_centers'}

    dist, _, _, h, o = min(candidates, key=lambda c: c[:3])
    valid = dist <= max_distance
    return {
        'valid': int(valid),
        'reason': 'matched' if valid else 'centers_too_far',
        'hand_score': float(h['score']),
        'object_score': float(o['score']),
        'hand_id': h.get('hand_id', ''),
        'object_id': o.get('object_id', ''),
        'hand_cx': float(h['cx']),
        'hand_cy': float(h['cy']),
        'obj_cx': float(o['cx']),
        'obj_cy': float(o['cy']),
        'match_distance_px': round(dist, 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left-dir')
    p.add_argument('--right-dir')
    p.add_argument('--output', required=True)
    p.add_argument('--pick-camera-1', choices=['left', 'right', 'front'], default='left')
    p.add_argument('--pick-camera-2', choices=['left', 'right', 'front'], default='right')
    p.add_argument('--work-dir', type=Path, default=None)
    p.add_argument('--num-frames', type=int, default=61)
    p.add_argument('--hand-thresh', type=float, default=0.5)
    p.add_argument('--object-thresh', type=float, default=0.5)
    p.add_argument('--max-match-distance-px', type=float, default=40.0)
    a = p.parse_args()
    if a.pick_camera_1 == a.pick_camera_2:
        p.error('Selezionare due camere differenti')
    if a.max_match_distance_px < 0:
        p.error('--max-match-distance-px non può essere negativo')

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    def folder_for(camera):
        if camera == 'left' and a.left_dir:
            return Path(a.left_dir)
        if camera == 'right' and a.right_dir:
            return Path(a.right_dir)
        if a.work_dir:
            return a.work_dir / camera
        return out / camera

    camera_dirs = [folder_for(a.pick_camera_1), folder_for(a.pick_camera_2)]
    datasets = [
        (load(d / 'hand_detections.csv'), load(d / 'object_detections.csv'))
        for d in camera_dirs
    ]
    rows = []
    pick = None
    for t in range(a.num_frames):
        first = inspect(t, *datasets[0], a.hand_thresh, a.object_thresh, a.max_match_distance_px)
        second = inspect(t, *datasets[1], a.hand_thresh, a.object_thresh, a.max_match_distance_px)
        both = int(bool(first['valid'] and second['valid']))
        if pick is None and both:
            pick = t
        # Conserva prefissi left/right: process_human_osvi.py già legge queste colonne.
        row = {
            'frame': f'frame_{t:05d}',
            'camera_1': a.pick_camera_1,
            'camera_2': a.pick_camera_2,
            'left_valid': first['valid'],
            'right_valid': second['valid'],
            'simultaneous': both,
        }
        row.update({'left_' + k: v for k, v in first.items() if k != 'valid'})
        row.update({'right_' + k: v for k, v in second.items() if k != 'valid'})
        rows.append(row)

    base_cols = ['frame', 'camera_1', 'camera_2', 'left_valid', 'right_valid', 'simultaneous']
    cols = base_cols + list(dict.fromkeys(k for row in rows for k in row if k not in base_cols))
    with (out / 'pick_matched_timeline.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    result = {
        'found': pick is not None,
        'pick_frame': pick,
        'num_frames': a.num_frames,
        'criterion': 'both cameras: right hand + portable state + detected object with bbox center distance',
        'match_method': 'bbox_centers',
        'hand_thresh': a.hand_thresh,
        'object_thresh': a.object_thresh,
        'max_match_distance_px': a.max_match_distance_px,
        'aspect_ratio_filter': False,
        'pick_camera_1': a.pick_camera_1,
        'pick_camera_2': a.pick_camera_2,
    }
    (out / 'pick_matched_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'[PICK] frame={pick}' if pick is not None else '[PICK] Nessun frame valido')
    print('[MATCH METHOD] bbox centers')
    print('[OUTPUT]', out / 'pick_matched_result.json')
    print('[TIMELINE]', out / 'pick_matched_timeline.csv')


if __name__ == '__main__':
    main()
