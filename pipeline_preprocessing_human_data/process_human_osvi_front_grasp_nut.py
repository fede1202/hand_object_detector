#!/usr/bin/env python3
"""Human PKL -> OSVI PKL: LEFT picks n, logical grasp n+offset; FRONT RGB-D trajectory.
Front object bbox at t in [n,n+offset] if associated to right hand, otherwise
nearest valid front hand + configurable BASE-frame XYZ offset. No LEFT->FRONT extrinsics.
Run on trusted local PKLs; geometry preparation ONLY, NOT robot motion commands.
"""
import argparse
import csv
import importlib.util
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def module_from_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(cmd, cwd=None):
    print('[RUN]', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


def frame_idx(s):
    m = re.search(r'(\d+)$', Path(str(s)).stem)
    if not m:
        raise ValueError(f'Frame name invalid: {s!r}')
    return int(m.group(1))


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def extract_images(data, out, n):
    camera_keys = {
        'front': 'camera_front_image',
        'left': 'camera_lateral_left_image',
            }
    for camera in camera_keys:
        (out / 'frames' / camera).mkdir(parents=True, exist_ok=True)
    traj = data['traj']
    for t in range(n):
        obs = traj.get(t)['obs']
        for camera, key in camera_keys.items():
            raw = np.asarray(obs[key])
            if raw.ndim == 3 and raw.shape[2] == 3:
                if camera == "front":
                    bgr = raw.copy()
                else:
                    bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            elif raw.ndim in (1, 2) and (raw.ndim == 1 or raw.shape[1] == 1):
                bgr = cv2.imdecode(raw.astype(np.uint8).reshape(-1), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f'Unable to decode image frame={t} key={key}')
            else:
                raise ValueError(f'Unsupported image shape {raw.shape}: {key} frame={t}')
            path = out / 'frames' / camera / f'frame_{t:05d}.png'
            if not cv2.imwrite(str(path), bgr):
                raise RuntimeError(f'Could not save {path}')


# ROI coordinates are in the ORIGINAL 672x376 pixel coordinate system.
# Negative/out-of-bounds inputs are clipped to the image dimensions.
DEFAULT_ROIS = {
    'front': [93, 63, 648, 368],
    'left': [0, -1, 668, 333],
    'right': [120, 1, 665, 374],
}

ROI_POLYGONS = {
    'left': np.array([
        [65, 192],
        [315, 18],
        [610, 85],
        [470, 333],
        [105, 333],
    ], dtype=np.float32)
}


def filter_detections_by_roi(work_dir, rois, image_width=672, image_height=376):
    """
    Filtra le detection mantenendo invariati i CSV originali.

    Camera LEFT:
        - Mano: centro bbox dentro la ROI poligonale.
        - Oggetto: intera bbox dentro la ROI poligonale.

    Camera FRONT:
        - Mano: centro bbox dentro la ROI rettangolare.
        - Oggetto: intera bbox dentro la ROI rettangolare.
    """

    target = work_dir / 'roi_filtered'
    summary = {}

    for camera in ('front', 'left'):

        # Determina il tipo di ROI della camera.
        use_polygon = camera in ROI_POLYGONS

        if use_polygon:
            polygon = np.asarray(
                ROI_POLYGONS[camera],
                dtype=np.float32
            ).reshape(-1, 2)

            if len(polygon) < 3 or not np.isfinite(polygon).all():
                raise ValueError(f'Invalid polygon for {camera}')

            polygon = np.ascontiguousarray(polygon)

            def inside(u, v):
                return cv2.pointPolygonTest(
                    polygon,
                    (float(u), float(v)),
                    False
                ) >= 0

            roi_description = {
                'type': 'polygon',
                'vertices': polygon.tolist()
            }

        else:
            x1, y1, x2, y2 = map(float, rois[camera])

            x1 = max(0., min(float(image_width), x1))
            x2 = max(0., min(float(image_width), x2))
            y1 = max(0., min(float(image_height), y1))
            y2 = max(0., min(float(image_height), y2))

            if not (x1 < x2 and y1 < y2):
                raise ValueError(
                    f'Invalid rectangular ROI for {camera}: {rois[camera]}'
                )

            def inside(u, v):
                return x1 <= float(u) < x2 and y1 <= float(v) < y2

            roi_description = {
                'type': 'rectangle',
                'coordinates': [x1, y1, x2, y2]
            }

        dst = target / camera
        dst.mkdir(parents=True, exist_ok=True)

        summary[camera] = {'roi': roi_description}

        for filename in ('hand_detections.csv', 'object_detections.csv'):

            src = work_dir / camera / filename

            with src.open(newline='') as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames

                if not {'cx', 'cy'}.issubset(fields or []):
                    raise ValueError(f'Missing center columns in {src}')

                rows = list(reader)

            kept = []

            for row in rows:
                try:

                    if filename == 'hand_detections.csv':
                        # MANO: controlla solo il centro.
                        u = float(row['cx'])
                        v = float(row['cy'])

                        valid = (
                            np.isfinite([u, v]).all()
                            and inside(u, v)
                        )

                    elif filename == 'object_detections.csv':
                        # OGGETTO: tutti i vertici della bbox dentro la ROI.
                        bx1 = float(row['x1'])
                        by1 = float(row['y1'])
                        bx2 = float(row['x2'])
                        by2 = float(row['y2'])

                        valid = (
                            np.isfinite([bx1, by1, bx2, by2]).all()
                            and bx1 < bx2
                            and by1 < by2
                            and inside(bx1, by1)
                            and inside(bx2, by1)
                            and inside(bx2, by2)
                            and inside(bx1, by2)
                        )

                    else:
                        valid = False

                    if valid:
                        kept.append(row)

                except (ValueError, TypeError, KeyError):
                    continue

            # Salva i CSV filtrati.
            with (dst / filename).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(kept)

            summary[camera][filename] = {
                'total': len(rows),
                'kept': len(kept)
            }

            print(
                f'[ROI] {camera} {filename}: '
                f'{len(kept)}/{len(rows)} kept',
                flush=True
            )

    (target / 'roi_filter_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n'
    )

    return target


def get_match_at_t(pick_mod, t, directory, args):
    hands = pick_mod.load(directory / 'hand_detections.csv')
    objs = pick_mod.load(directory / 'object_detections.csv')
    return pick_mod.inspect(t, hands, objs, args.hand_thresh,
                            args.object_thresh, args.max_match_distance_px)


def point3d(recon, depth, u, v, args):
    d, frac, count, status = recon.median_patch(
        depth, u, v, args.depth_patch // 2, args.min_valid_depth,
        args.min_depth, args.max_depth)
    if status != 'ok':
        return np.full(3, np.nan), status
    v_cam = np.array([(u - recon.CX) * d / recon.FX,
                      (v - recon.CY) * d / recon.FY,
                      d, 1.0], dtype=np.float64)
    xyz = (recon.T_BASE_CAMERA @ v_cam)[:3]
    return xyz, 'ok' if np.isfinite(xyz).all() else 'invalid_projection'


def fill_gaps(xyz):
    valid = np.flatnonzero(np.isfinite(xyz).all(axis=1))
    if not len(valid):
        raise ValueError('No valid right-hand 3D point. Check detections and front depth.')
    if len(valid) == 1 and len(xyz) > 1:
        raise ValueError('Only one valid hand point; cannot interpolate trajectory.')
    missing = np.flatnonzero(~np.isfinite(xyz).all(axis=1)).tolist()
    res = xyz.copy()
    for k in range(3):
        res[:, k] = np.interp(np.arange(len(res)), valid, xyz[valid, k])
    return res, missing


# Fixed top-down orientation from the OSVI-AWDA controller configuration.
TOP_DOWN_QUAT = np.array([0., 0., 1., 0.], dtype=np.float64)


def print_pick_diagnostics(timeline_file, cameras=('left', 'right'), max_rows=12):
    rows = read_csv(timeline_file)
    print('[DIAGNOSTICS] Pick non rilevato. Motivi per camera:', flush=True)
    from collections import Counter
    for side in cameras:
        counts = Counter(r.get(f'{side}_reason', '') for r in rows)
        print(' ', side, dict(counts), flush=True)
    print('[DIAGNOSTICS] Frame con almeno una camera valida:', flush=True)
    candidates = [r for r in rows if r.get('left_valid') == '1' or r.get('right_valid') == '1']
    for r in candidates[:max_rows]:
        print(' ', r.get('frame'),
              f'{cameras[0]}=', r.get('left_reason'),
              f'{cameras[1]}=', r.get('right_reason'),
              f'{cameras[0]}_dist=', r.get('left_match_distance_px', ''),
              f'{cameras[1]}_dist=', r.get('right_match_distance_px', ''), flush=True)
    if not candidates:
        print('  Nessun frame con una singola camera valida.', flush=True)

# def inspect_left_geometry(t, hands, objects, args):
#     """
#     Verifica tutte le coppie mano-oggetto nel frame t.

#     Mano:
#       - destra (side=1)
#       - Portable Object (contact_state=3)
#       - confidence >= hand_thresh

#     Oggetto:
#       - confidence >= object_thresh

#     Geometria:
#       - distanza <= max_match_distance_px
#       - dx < 0
#       - dy > 0
#       - slope < left_max_slope
#     """

#     best = None

#     for hand in hands.get(t, []):
#         try:
#             if int(float(hand['side'])) != 1:
#                 continue
#             if int(float(hand['contact_state'])) != 3:
#                 continue
#             if float(hand['score']) < args.hand_thresh:
#                 continue

#             hx = float(hand['cx'])
#             hy = float(hand['cy'])

#             for obj in objects.get(t, []):
#                 if float(obj['score']) < args.object_thresh:
#                     continue

#                 ox = float(obj['cx'])
#                 oy = float(obj['cy'])

#                 dx = ox - hx
#                 dy = oy - hy

#                 if not np.isfinite([dx, dy]).all():
#                     continue

#                 distance = float(np.hypot(dx, dy))

#                 if distance > args.max_match_distance_px:
#                     continue

#                 if dx >= -1e-6 or dy <= 0:
#                     continue

#                 slope = dy / dx

#                 if slope >= args.left_max_slope:
#                     continue

#                 candidate = {
#                     'geometry_valid': True,
#                     'geometry_distance_px': distance,
#                     'geometry_dx': dx,
#                     'geometry_dy': dy,
#                     'geometry_slope': slope,
#                     'geometry_hand_id': hand.get('hand_id', ''),
#                     'geometry_object_id': obj.get('object_id', '')
#                 }

#                 if best is None or distance < best['geometry_distance_px']:
#                     best = candidate

#         except (ValueError, TypeError, KeyError, ZeroDivisionError):
#             continue

#     if best is None:
#         return {
#             'geometry_valid': False,
#             'geometry_distance_px': '',
#             'geometry_dx': '',
#             'geometry_dy': '',
#             'geometry_slope': '',
#             'geometry_hand_id': '',
#             'geometry_object_id': ''
#         }

#     return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pkl', required=True, type=Path, help='Trusted original human PKL')
    p.add_argument('--output-pkl', required=True, type=Path)
    p.add_argument('--work-dir', required=True, type=Path)
    p.add_argument('--repo', required=True, type=Path, help='Existing hand_object_detector directory')
    p.add_argument('--utils', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--open-value', type=float, required=True, help='Actual action[-1] value for open gripper')
    p.add_argument('--closed-value', type=float, required=True, help='Actual action[-1] value for closed gripper')
    p.add_argument('--hand-thresh', type=float, default=0.5)
    p.add_argument('--object-thresh', type=float, default=0.5)
    p.add_argument('--max-match-distance-px', type=float, default=40)
    p.add_argument(
        '--left-max-slope',
        type=float,
        default=-0.4,
        help='Pendenza massima mano-oggetto LEFT in coordinate immagine'
    )
    p.add_argument('--depth-patch', type=int, default=7)
    p.add_argument(
        '--grasp-offset-u-px',
        type=float,
        default=0.0,
        help='Offset orizzontale in pixel del grasp FRONT prima della depth'
    )

    p.add_argument(
        '--grasp-offset-v-px',
        type=float,
        default=0.0,
        help='Offset verticale in pixel del grasp FRONT prima della depth'
    )
    p.add_argument('--min-valid-depth', type=float, default=0.5)
    p.add_argument('--min-depth', type=float, default=0.15)
    p.add_argument('--max-depth', type=float, default=2.5)
    p.add_argument('--pick-delay', type=int, default=2, help='Logical grasp frame = first left detection + delay (default: 2)')
    p.add_argument('--grasp-offset-xyz', type=float, nargs=3, metavar=('DX','DY','DZ'), default=[0.,0.,0.], help='Hand fallback offset in base_link [meters]; calibrate before robot use')
    p.add_argument('--front-object-max-distance-px', type=float, default=40., help='Max center separation of object and right hand on FRONT in pick window')
    p.add_argument('--front-object-min-score', type=float, default=None, help='Default to --object-thresh')
    p.add_argument('--skip-object-grasp', action='store_true', help='Always take front hand fallback; useful for baseline')
    p.add_argument('--roi-json', type=Path, default=None, help='Optional JSON with front/left/right ROIs; defaults to the selected coordinates')
    p.add_argument('--skip-detection', action='store_true', help='Use existing CSVs in work-dir')
    p.add_argument('--pick-frame', type=int, default=None, help='Manual override of LEFT detected frame n (delay still applied)')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if a.pick_delay < 0:
        p.error('--pick-delay must be >= 0')
    if a.front_object_max_distance_px < 0:
        p.error('--front-object-max-distance-px must be >= 0')
    if not np.isfinite(a.grasp_offset_xyz).all():
        p.error('--grasp-offset-xyz must contain finite numbers')
    if a.pkl.resolve() == a.output_pkl.resolve():
        p.error('Input and output PKL must be different')
    if a.output_pkl.exists() and not a.overwrite:
        p.error('Output already exists; use --overwrite')
    if a.open_value == a.closed_value:
        p.error('Open and closed must differ')
    if a.depth_patch <= 0 or a.depth_patch % 2 == 0:
        p.error('--depth-patch must be positive and odd')
    for script in ('filter_pick_matched_object.py', 'reconstruct_right_hand_3d.py'):
        if not (a.utils / script).is_file():
            p.error(f'Missing existing helper: {a.utils / script}')
    if not (a.repo / 'demo.py').is_file():
        p.error(f'Missing {a.repo / "demo.py"}')

    a.work_dir.mkdir(parents=True, exist_ok=True)
    with a.pkl.open('rb') as f:
        data = pickle.load(f)
    traj = data['traj']
    n = len(traj)
    if not hasattr(traj, '_data') or len(traj._data) != n:
        raise TypeError('Expected original human Trajectory with _data')
    print('[INPUT]', a.pkl, 'frames', n, flush=True)
    if not a.skip_detection:
        extract_images(data, a.work_dir, n)
        for camera in ('left', 'front'):
            run([sys.executable, 'demo.py', '--net', 'res101', '--dataset', 'pascal_voc',
                 '--load_dir', 'models', '--checksession', '1', '--checkepoch', '8',
                 '--checkpoint', '89999', '--cuda',
                 '--thresh_hand', str(a.hand_thresh), '--thresh_obj', str(a.object_thresh),
                 '--image_dir', a.work_dir / 'frames' / camera,
                 '--save_dir', a.work_dir / camera], cwd=a.repo)

    for camera in ('left', 'front'):
        for name in ('hand_detections.csv', 'object_detections.csv'):
            path = a.work_dir / camera / name
            if not path.is_file():
                raise FileNotFoundError(f'Expected {path}. Check demo.py exporter.')

    if a.roi_json is None:
        rois = DEFAULT_ROIS
    else:
        rois = json.loads(a.roi_json.read_text())
    if not {'front', 'left'}.issubset(rois):
        raise ValueError('ROI JSON needs front and left')
    filtered_dir = filter_detections_by_roi(a.work_dir, rois)

    # Detect n on LEFT only; uses original Portable right hand + near object criterion.
    pick_mod = module_from_file('original_pick_filter', a.utils / 'filter_pick_matched_object.py')
    left_hands = pick_mod.load(filtered_dir / 'left' / 'hand_detections.csv')
    left_objects = pick_mod.load(filtered_dir / 'left' / 'object_detections.csv')
    timeline = []
    first_pick = None

    for t in range(n):

        inspection = pick_mod.inspect(
            t,
            left_hands,
            left_objects,
            a.hand_thresh,
            a.object_thresh,
            a.max_match_distance_px
        )

        # geometry = inspect_left_geometry(
        #     t,
        #     left_hands,
        #     left_objects,
        #     a
        # )

        # Il pick deve rispettare sia il criterio originale
        # sia la geometria della coppia mano-oggetto.
        valid_pick = bool(
            inspection['valid'] #and geometry['geometry_valid']
        )

        if first_pick is None and valid_pick:
            first_pick = t

        timeline.append({
            'frame': f'frame_{t:05d}',
            'camera': 'left',
            **inspection,
            #**geometry,
            'valid_after_geometry': int(valid_pick)
        })
    cols = ['frame', 'camera'] + list(dict.fromkeys(
        k for row in timeline for k in row if k not in ('frame', 'camera')))
    with (a.work_dir / 'pick_matched_timeline.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(timeline)
    result = {
        'found': first_pick is not None,
        'pick_frame': first_pick,
        'pick_camera': 'left',
        'criterion': 'right hand portable + nearby object + slope filter',
        'max_match_distance_px': a.max_match_distance_px,
        #'left_max_slope': a.left_max_slope,
        #'geometry_rule': 'dx < 0 and dy > 0 and slope < left_max_slope'
    }
    (a.work_dir / 'pick_matched_result.json').write_text(json.dumps(result, indent=2) + '\n')
    if first_pick is None and a.pick_frame is None:
        from collections import Counter
        print('[LEFT PICK DIAGNOSTICS]', dict(Counter(row['reason'] for row in timeline)))
        raise ValueError('No pick detected on left: see pick_matched_timeline.csv')
    detected_pick = int(a.pick_frame if a.pick_frame is not None else first_pick)
    pick = detected_pick + a.pick_delay
    if not (0 <= detected_pick < n and 0 < pick < n - 1):
        raise ValueError(f'Invalid pick n={detected_pick}, delayed pick={pick}, length={n}. '
                         'Need at least one frame after delayed grasp for loader.')
    print(f'[PICK] left detection n={detected_pick}; logical grasp n+{a.pick_delay}={pick}')

    recon = module_from_file('original_reconstruction', a.utils / 'reconstruct_right_hand_3d.py')
    front_hands = pick_mod.load(filtered_dir / 'front' / 'hand_detections.csv')
    xyz = np.full((n, 3), np.nan, dtype=np.float64)
    diagnostics = []
    for t in range(n):
        obs = traj.get(t)['obs']
        depth = np.asarray(obs['camera_front_depth'])
        if depth.shape != (recon.H, recon.W):
            raise ValueError(f'frame {t} depth: shape {depth.shape}')
        hands = [h for h in front_hands.get(t, []) if int(float(h['side'])) == 1
                 and float(h['score']) >= a.hand_thresh]
        if hands:
            h = max(hands, key=lambda v: float(v['score']))
            u, v = float(h['cx']), float(h['cy'])
            xyz[t], status = point3d(recon, depth, u, v, a)
        else:
            u = v = None
            status = 'missing_right_hand'
        diagnostics.append(dict(frame=f'frame_{t:05d}', u=u, v=v,
                                x=xyz[t, 0], y=xyz[t, 1], z=xyz[t, 2], status=status,
                                source='front_hand'))

    positions, interpolated = fill_gaps(xyz)
    # GRASP SOURCE: object FRONT near right-hand FRONT in frames n..pick; prefer
    # frame nearest pick (descending). Every object must have valid front depth.
    # Since generic detector has no color/instance identity, this is a proximity
    # hypothesis only; inspect grasp_candidates.csv and overlay before training.
    front_objects = pick_mod.load(filtered_dir / 'front' / 'object_detections.csv')
    offset = np.asarray(a.grasp_offset_xyz, dtype=np.float64)
    object_threshold = a.object_thresh if a.front_object_min_score is None else a.front_object_min_score
    candidates_log = []
    selected = None
    if not a.skip_object_grasp:
        for t in range(pick, detected_pick - 1, -1):
            obs = traj.get(t)['obs']
            depth = np.asarray(obs['camera_front_depth'])
            hand = diagnostics[t]
            for obj in front_objects.get(t, []):
                log = {'frame': t, 'object_id': obj.get('object_id', ''),
                       'object_score': obj.get('score', ''), 'obj_u': obj.get('cx', ''),
                       'obj_v': obj.get('cy', ''), 'hand_u': hand['u'],
                       'hand_v': hand['v'], 'distance_px': '',
                       'status': 'unverified', 'x': '', 'y': '', 'z': ''}
                try:
                    score = float(obj['score'])
                    u, v = float(obj['cx']), float(obj['cy'])
                    if not np.isfinite([score, u, v]).all() or score < object_threshold:
                        log['status'] = 'low_score_or_invalid'
                    elif hand['u'] is None or hand['v'] is None or hand['status'] != 'ok':
                        log['status'] = 'no_valid_front_hand_association'
                    else:
                        # Distanza mano-oggetto calcolata sui centri originali
                        distance = float(np.hypot(
                            u - hand['u'],
                            v - hand['v']
                        ))

                        log['distance_px'] = distance

                        if distance > a.front_object_max_distance_px:
                            log['status'] = 'too_far_from_front_hand'

                        else:
                            # Applica offset al centro dell'oggetto FRONT
                            grasp_u = u + a.grasp_offset_u_px
                            grasp_v = v + a.grasp_offset_v_px

                            # Calcola la profondità nella NUOVA posizione
                            # e ricostruisce il punto nel frame base_link
                            point, status = point3d(
                                recon,
                                depth,
                                grasp_u,
                                grasp_v,
                                a
                            )

                            log['status'] = status
                            if status == 'ok':
                                log.update(x=float(point[0]), y=float(point[1]), z=float(point[2]))
                                # First prefer latest frame; within frame nearest eligible object.
                                if selected is None or t > selected['frame'] or (
                                    t == selected['frame'] and distance < selected['distance_px']):
                                    selected = {
                                        'frame': t,
                                        'point': point,
                                        'pixel': [grasp_u, grasp_v],
                                        'original_object_pixel': [u, v],
                                        'distance_px': distance,
                                        'score': score,
                                        'object_id': obj.get('object_id', '')
                                    }
                except (ValueError, TypeError, KeyError):
                    log['status'] = 'invalid_detection'
                candidates_log.append(log)
    with (a.work_dir / 'grasp_candidates.csv').open('w', newline='') as f:
        fields = ['frame','object_id','object_score','obj_u','obj_v','hand_u',
                  'hand_v','distance_px','status','x','y','z']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(candidates_log)

    if selected is not None:
        grasp_xyz = np.asarray(selected['point'], dtype=np.float64)
        grasp_source = 'front_object_near_hand_unverified_identity'
        grasp_source_frame = int(selected['frame'])
        grasp_uv = selected['pixel']
        object_match = {'distance_px': selected['distance_px'],
                        'score': selected['score'], 'object_id': selected['object_id']}
    else:
        # Only use direct, valid hand detections (NOT filled/interpolated points).
        valid_hand = np.flatnonzero(np.isfinite(xyz).all(axis=1))
        if not len(valid_hand):
            raise ValueError('No valid FRONT hand with depth; cannot build grasp')
        # Tie-break: earlier frames first, to avoid picking a post-grasp hand pose.
        nearest = min((int(t) for t in valid_hand), key=lambda t: (abs(t-pick), t > pick, t))
        grasp_xyz = xyz[nearest] + offset
        grasp_source_frame = nearest
        grasp_uv = [diagnostics[nearest]['u'], diagnostics[nearest]['v']]
        grasp_source = 'front_hand_at_pick_offset' if nearest == pick else 'nearest_valid_front_hand_offset'
        object_match = None
        if not np.any(offset):
            print('[WARNING] grasp hand offset is (0,0,0): calibrate XYZ before robot use')

    # Replace ONLY grasp waypoint, keeping all other FRONT-hand 3D positions.
    hand_xyz_at_pick = positions[pick].copy()
    positions[pick] = grasp_xyz
    grasp_jump_prev = float(np.linalg.norm(positions[pick] - positions[pick-1]))
    grasp_jump_next = float(np.linalg.norm(positions[pick+1] - positions[pick]))
    if max(grasp_jump_prev, grasp_jump_next) > 0.08:
        print(f'[WARNING] grasp replacement creates adjacent step '
              f'{grasp_jump_prev:.3f} / {grasp_jump_next:.3f} m; verify trajectory')
    diagnostics[pick]['source'] = grasp_source
    print(f'[GRASP] logical_frame={pick}, source_frame={grasp_source_frame}, '
          f'source={grasp_source}, XYZ={grasp_xyz}')

    # Inspection overlay on the matching FRONT frame (not frame zero).
    raw_front = np.asarray(traj.get(grasp_source_frame)['obs']['camera_front_image']).copy()
    if raw_front.ndim == 3 and raw_front.shape[2] == 3:
        for label, pixel, color in [('GRASP SOURCE', grasp_uv, (0, 255, 0))]:
            if pixel is not None and all(v is not None for v in pixel):
                x, y = (int(round(v)) for v in pixel)
                cv2.drawMarker(raw_front, (x, y), color, cv2.MARKER_CROSS, 20, 2)
                cv2.putText(raw_front, f'{label} t={grasp_source_frame}',
                            (max(x-100, 0), max(y-12, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(str(a.work_dir / 'grasp_front_source_overlay.png'), raw_front)

    with (a.work_dir / 'final_waypoints.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame','hand_x','hand_y','hand_z',
                                           'final_x','final_y','final_z','is_grasp',
                                           'hand_detection_status'])
        w.writeheader()
        for t in range(n):
            w.writerow({'frame': t,
                        'hand_x': xyz[t, 0], 'hand_y': xyz[t, 1], 'hand_z': xyz[t, 2],
                        'final_x': positions[t, 0], 'final_y': positions[t, 1],
                        'final_z': positions[t, 2], 'is_grasp': int(t == pick),
                        'hand_detection_status': diagnostics[t]['status']})

    quat = TOP_DOWN_QUAT.copy()

    # Stato gripper associato direttamente a ogni frame reale:
    # - prima del pick: aperto
    # - dal pick fino al penultimo frame: chiuso
    # - ultimo frame reale: aperto
    action_gripper = np.full(n, a.open_value, dtype=float)

    action_gripper[pick:n - 1] = a.closed_value

    action_gripper[-1] = a.open_value
    for t in range(n):
        old = traj._data[t]
        if not (isinstance(old, tuple) and len(old) == 5):
            raise TypeError(f'Unexpected Trajectory._data[{t}] layout')
        obs, reward, done, info, unused_action = old
        obs_new = dict(obs)
        obs_new['eef_pos'] = np.asarray(positions[t], dtype=np.float64)
        action = np.concatenate([positions[t], quat, [action_gripper[t]]]).astype(np.float64)
        traj._data[t] = (obs_new, reward, done, info, action)

    # Preserve original _raw_state, _config_str and top-level metadata.
    a.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with a.output_pkl.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with a.output_pkl.open('rb') as f:
        loaded = pickle.load(f)
    assert len(loaded['traj']) == n
    for t in range(n):
        assert np.asarray(loaded['traj'].get(t)['obs']['eef_pos']).shape == (3,)
        assert np.asarray(loaded['traj'].get(t)['action']).shape == (8,)

    with (a.work_dir / 'reconstruction_diagnostics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame', 'u', 'v', 'x', 'y', 'z', 'status', 'source'])
        w.writeheader()
        w.writerows(diagnostics)
    info = dict(input=str(a.pkl), output=str(a.output_pkl), n=n,
                detected_pick_frame=detected_pick, pick_delay=a.pick_delay,
                pick_frame=pick, grasp_source_frame=grasp_source_frame,
                grasp_position_source=grasp_source, grasp_xyz_base=grasp_xyz.tolist(),
                grasp_source_pixel_front=grasp_uv, 
                grasp_pixel_offset_uv=[
                    a.grasp_offset_u_px,
                    a.grasp_offset_v_px
                ], 
                front_object_association=object_match,
                front_object_identity_verified=False,
                hand_xyz_base_at_pick_before_replacement=hand_xyz_at_pick.tolist(),
                grasp_offset_base_m=offset.tolist(),
                adjacent_step_before_grasp_m=grasp_jump_prev,
                adjacent_step_after_grasp_m=grasp_jump_next,
                interpolated_hand_frames=interpolated, quaternion=quat.tolist(),
                open_value=a.open_value, closed_value=a.closed_value,
                last_action_open=True,
                gripper_associated_to_same_frame=True,
                pick_cameras=['left'], trajectory_camera='front')
    (a.work_dir / 'human_osvi_build_report.json').write_text(json.dumps(info, indent=2) + '\n')
    print('[DONE] PKL', a.output_pkl)
    print('[PICK]', pick, 'grasp XYZ', grasp_xyz)
    print('[GRIPPER] Open before pick, closed from pick to penultimate frame, open on last real frame.')


if __name__ == '__main__':
    main()
