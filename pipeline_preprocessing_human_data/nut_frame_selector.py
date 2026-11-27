#!/usr/bin/env python3
"""Nut Assembly frame selector: stdlib web server + OpenCV thumbnails."""
import argparse
import csv
import io
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

HTML = r'''<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nut Assembly · selezione ultimo frame</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#121720;color:#f0f4fb}header{position:sticky;top:0;z-index:5;background:#202938;border-bottom:1px solid #364254;padding:12px 18px;display:flex;align-items:center;flex-wrap:wrap;gap:10px}h2{margin:0 12px 0 0;font-size:18px}button,select,input{background:#303c50;color:#fff;border:1px solid #516178;border-radius:7px;padding:8px 10px;font-size:14px}button{cursor:pointer}button:hover{background:#40536d}.primary{background:#08785c;border-color:#129b79;font-weight:700}.primary:hover{background:#0b906e}.layout{display:grid;grid-template-columns:minmax(230px,300px) 1fr;min-height:calc(100vh - 75px)}aside{padding:12px;background:#19212d;border-right:1px solid #374254;max-height:calc(100vh - 75px);overflow:auto;position:sticky;top:75px}.entry{display:block;width:100%;text-align:left;margin-bottom:5px;padding:9px;border-radius:7px}.entry.active{border-color:#77b6ff;background:#263d5b}.entry.done:before{content:'✓ ';color:#3be0a7}main{padding:16px;min-width:0}.info{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}#preview{max-width:100%;max-height:47vh;object-fit:contain;background:#05080b;border-radius:8px;border:1px solid #3b4859}#gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:10px;margin-top:15px}.tile{display:block;border:2px solid #37445b;border-radius:8px;padding:4px;background:#1e2937;cursor:pointer;text-align:center}.tile img{width:100%;aspect-ratio:672/376;object-fit:contain;background:#070c10}.tile span{display:block;padding:4px}.tile.selected{border-color:#31d6a2;box-shadow:0 0 0 2px #31d6a244}.notice{color:#c6d2e0;font-size:13px}.saving{color:#ffd383}#status{font-size:14px;font-weight:600}#search{width:100%;margin:8px 0 12px}@media(max-width:750px){.layout{grid-template-columns:1fr}aside{position:relative;top:auto;max-height:210px;border-right:0;border-bottom:1px solid #374254}header{position:relative}#gallery{grid-template-columns:repeat(auto-fill,minmax(135px,1fr))}}
</style></head><body>
<header><h2>Nut Assembly · ultimo frame</h2><button id="prev">← Precedente</button><button id="next">Successiva →</button><select id="camera"></select><button class="primary" id="save">Salva selezione nel CSV</button><span id="status"></span></header>
<div class="layout"><aside><div id="progress" class="notice"></div><input id="search" placeholder="Filtra task o traiettoria"><div id="entries"></div></aside><main><div class="info"><strong id="title">Caricamento…</strong><span id="counts"></span><span id="chosen"></span></div><div><img id="preview" alt="Frame selezionato"></div><div class="notice">Clicca un frame per indicare l'ultimo da mantenere. Frecce ← → per spostarti tra i frame; premi Salva per registrare la scelta. La camera serve solo per la visualizzazione: il taglio riguarda la traiettoria intera.</div><div id="gallery"></div></main></div>
<script>
let items=[],idx=0,selected=null,unsaved=false,filter='';
const el=id=>document.getElementById(id);
async function api(url,options){const r=await fetch(url,options);const x=await r.json();if(!r.ok)throw Error(x.error||'Errore richiesta');return x}
function item(){return items[idx]}
function qImg(which,frame){const x=item();return `/${which}?task=${encodeURIComponent(x.task)}&traj=${encodeURIComponent(x.traj)}&camera=${encodeURIComponent(el('camera').value)}&frame=${frame}`}
function setSelected(n){selected=n;unsaved=true;el('chosen').textContent=`Ultimo frame: ${n} · mantenere ${n+1} frame (non salvato)`;el('status').textContent='Selezione da salvare';el('status').className='saving';el('preview').src=qImg('image',n);document.querySelectorAll('.tile').forEach(b=>b.classList.toggle('selected',Number(b.dataset.frame)===n))}
function renderList(){let count=items.filter(x=>x.saved!==null).length;el('progress').textContent=`Annotate ${count} / ${items.length} traiettorie`;el('entries').innerHTML='';items.forEach((x,i)=>{if(!`${x.task}/${x.traj}`.toLowerCase().includes(filter))return;const b=document.createElement('button');b.className='entry'+(i===idx?' active':'')+(x.saved!==null?' done':'');b.textContent=`${x.task}/${x.traj} · ${x.frames} frame${x.saved!==null?' · ultimo '+x.saved:''}`;b.onclick=()=>openItem(i);el('entries').appendChild(b)})}
function openItem(i){if(unsaved&&!confirm('La selezione corrente non è stata salvata. Proseguire?'))return;idx=i;let x=item();selected=x.saved;unsaved=false;el('camera').innerHTML='';for(const c of Object.keys(x.cameras)){const op=document.createElement('option');op.value=c;op.textContent=c;el('camera').appendChild(op)}if(x.cameras.camera_front_image)el('camera').value='camera_front_image';el('title').textContent=`${x.task} / ${x.traj}`;el('counts').textContent=`Frame originali: ${x.frames}`;el('chosen').textContent=selected===null?'Nessun ultimo frame selezionato':`Ultimo frame salvato: ${selected} · mantenere ${selected+1} frame`;el('status').textContent='';renderGallery();renderList()}
function renderGallery(){let x=item(),c=el('camera').value,frames=x.cameras[c]||[];el('gallery').innerHTML='';const frag=document.createDocumentFragment();frames.forEach(n=>{const b=document.createElement('button');b.className='tile'+(n===selected?' selected':'');b.dataset.frame=n;const im=document.createElement('img');im.loading='lazy';im.src=qImg('thumb',n);im.alt='Frame '+n;const s=document.createElement('span');s.textContent='Frame '+String(n).padStart(5,'0');b.append(im,s);b.onclick=()=>setSelected(n);frag.appendChild(b)});el('gallery').appendChild(frag);const show=(selected!==null&&frames.includes(selected))?selected:(frames[0]??null);if(show!==null)el('preview').src=qImg('image',show);else el('preview').removeAttribute('src')}
async function save(){if(selected===null){alert('Seleziona prima un frame');return}try{const x=item();const result=await api('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:x.task,traj:x.traj,last_frame:selected})});x.saved=selected;unsaved=false;el('chosen').textContent=`Ultimo frame salvato: ${selected} · mantenere ${selected+1} frame`;el('status').textContent='✓ CSV aggiornato';el('status').className='';renderList()}catch(e){alert(e.message)}}
el('save').onclick=save;el('camera').onchange=renderGallery;el('prev').onclick=()=>openItem(Math.max(0,idx-1));el('next').onclick=()=>openItem(Math.min(items.length-1,idx+1));el('search').oninput=e=>{filter=e.target.value.toLowerCase();renderList()};document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;const x=item();if(!x||selected===null)return;const frames=x.cameras[el('camera').value]||[];const k=frames.indexOf(selected);if(e.key==='ArrowRight'&&k<frames.length-1)setSelected(frames[k+1]);if(e.key==='ArrowLeft'&&k>0)setSelected(frames[k-1])});
(async()=>{try{items=await api('/api/list');if(!items.length){el('title').textContent='Nessuna traiettoria trovata';return}openItem(0)}catch(e){el('title').textContent='Errore: '+e.message}})();
</script></body></html>'''

parser = argparse.ArgumentParser()
parser.add_argument('--images', required=True, type=Path)
parser.add_argument('--csv', required=True, type=Path)
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', type=int, default=8765)
args = parser.parse_args()

if not args.images.is_dir():
    parser.error(f'Cartella immagini inesistente: {args.images}')

records = {}
for task_dir in sorted(args.images.glob('task_*')):
    if not task_dir.is_dir():
        continue
    for traj_dir in sorted(task_dir.glob('traj*')):
        if not traj_dir.is_dir():
            continue
        cameras = {}
        for cam_dir in traj_dir.iterdir():
            if cam_dir.is_dir() and cam_dir.name.startswith('camera_'):
                frames = {}
                for p in cam_dir.glob('frame_*.png'):
                    m = re.fullmatch(r'frame_(\d+)\.png', p.name)
                    if m:
                        frames[int(m.group(1))] = p
                if frames:
                    cameras[cam_dir.name] = frames
        if cameras:
            base = cameras.get('camera_front_image') or next(iter(cameras.values()))
            records[(task_dir.name, traj_dir.name)] = {'cameras': cameras, 'indices': sorted(base), 'count': max(base)+1}

if not records:
    parser.error('Nessun frame PNG trovato nel percorso specificato')

FIELDNAMES = ['task', 'trajectory', 'original_frames', 'last_frame', 'frames_to_keep']
lock = threading.Lock()
saved = {}
if args.csv.exists():
    with args.csv.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row.get('task'), row.get('trajectory'))
            try:
                if key in records:
                    v = int(row['last_frame'])
                    if v in records[key]['indices']:
                        saved[key] = v
            except (ValueError, TypeError, KeyError):
                pass

def write_csv():
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.csv.with_name(args.csv.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDNAMES)
        wr.writeheader()
        for (task, traj), last in sorted(saved.items()):
            wr.writerow({'task': task, 'trajectory': traj, 'original_frames': records[(task, traj)]['count'], 'last_frame': last, 'frames_to_keep': last+1})
    os.replace(tmp, args.csv)

class Handler(BaseHTTPRequestHandler):
    def respond(self, data, typ='application/json', status=200):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', typ)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store' if typ.startswith('application/json') else 'public, max-age=86400')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/':
            return self.respond(HTML.encode('utf-8'), 'text/html; charset=utf-8')
        if u.path == '/api/list':
            with lock:
                data = [{'task': t, 'traj': tr, 'frames': r['count'], 'cameras': {c: sorted(x) for c, x in r['cameras'].items()}, 'saved': saved.get((t, tr))} for (t, tr), r in records.items()]
            return self.respond(data)
        if u.path in ('/thumb', '/image'):
            q = parse_qs(u.query)
            def get(name): return q.get(name, [''])[0]
            key = (get('task'), get('traj'))
            r = records.get(key)
            try: n = int(get('frame'))
            except ValueError: return self.respond({'error': 'Frame non valido'}, status=400)
            p = r['cameras'].get(get('camera'), {}).get(n) if r else None
            if p is None: return self.respond({'error': 'Immagine non trovata'}, status=404)
            if u.path == '/image':
                return self.respond(p.read_bytes(), 'image/png')
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None: return self.respond({'error': 'Errore lettura immagine'}, status=500)
            h, w = img.shape[:2]
            img = cv2.resize(img, (min(270, w), int(h * min(270, w) / w)), interpolation=cv2.INTER_AREA)
            ok, data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 76])
            return self.respond(data.tobytes(), 'image/jpeg') if ok else self.respond({'error': 'JPEG fallito'}, status=500)
        return self.respond({'error': 'URL inesistente'}, status=404)

    def do_POST(self):
        if urlparse(self.path).path != '/api/save':
            return self.respond({'error': 'URL inesistente'}, status=404)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length > 10000: raise ValueError('Richiesta troppo grande')
            payload = json.loads(self.rfile.read(length))
            key = (payload['task'], payload['traj'])
            n = int(payload['last_frame'])
            if key not in records or n not in records[key]['indices']:
                raise ValueError('Traiettoria o frame non valido')
            with lock:
                saved[key] = n
                write_csv()
            return self.respond({'ok': True, 'last_frame': n})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            return self.respond({'error': str(e)}, status=400)
        except OSError as e:
            return self.respond({'error': f'Impossibile salvare CSV: {e}'}, status=500)

    def log_message(self, fmt, *args):
        if not self.path.startswith(('/thumb', '/image')):
            super().log_message(fmt, *args)

print(f'Traiettorie trovate: {len(records)}')
print(f'CSV: {args.csv}')
print(f'Apri nel browser: http://localhost:{args.port}')
ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
