# Pipeline Human Demonstration → OSVI-AWDA (UR5e)

## 1. Percorsi

```bash
export ROOT="/home/mivia/Desktop/UR-Application/Federica/hand_tracking_test"
export PYTHONPATH="/home/mivia/Desktop/UR-Application/Federica/Multi-Task-LFD-Training-Framework/training/multi_task_il:/home/mivia/Desktop/UR-Application/Federica/Multi-Task-LFD-Training-Framework/training:${PYTHONPATH:-}"
```

| Ruolo | Percorso relativo a `$ROOT` |
|---|---|
| Input, traiettorie ridotte | `dataset/human_reduced/task_XX/trajNNN.pkl` |
| Script principale | `utils/process_human_osvi_front_grasp.py` |
| Matching LEFT, helper | `utils/filter_pick_matched_object.py` |
| Geometria RGB-D e frame `base_link` | `utils/reconstruct_right_hand_3d.py` |
| Rete di rilevamento | `hand_object_detector/demo.py` |
| Plot delle traiettorie convertite | `utils/plot_osvi_pkl_on_front.py` |
| PKL definitivi | `dataset/human_osvi_front_grasp_final/task_XX/trajNNN.pkl` |
| Diagnostica completa per traiettoria | `front_grasp_final/task_XX/trajNNN/` |
| Log di elaborazione | `front_grasp_final_logs/task_XX-trajNNN.log` |
| Campione di sviluppo (16 PKL) | `front_grasp_test/selected_trajectories.txt` |


**Nota:** `process_human_osvi_front_grasp.py` importa dinamicamente entrambi i moduli helper con `importlib.util`. Il detector, quando abilitato, viene eseguito come sottoprocesso di `demo.py`; il plot è un comando **separato**, non uno step obbligatorio nel builder.

## 3. Configurazione finale consolidata

| Parametro | Valore |
|---|---|
| Dimensioni immagine FRONT / LEFT | **672 × 376** pixel |
| Camera per la decisione del pick | **LEFT** |
| Camera per la traiettoria 3D | **FRONT RGB-D** |
| Camera per la posizione oggetto/hand fallback | **FRONT RGB-D** |
| Detector mano / oggetto | Faster R-CNN Res101, dataset detector `pascal_voc` |
| Checkpoint detector in `demo.py` | `--checksession 1 --checkepoch 8 --checkpoint 89999` da `models/` |
| `--hand-thresh` | **0.5** |
| `--object-thresh` | **0.5** |
| Mano destra | `side == 1` |
| Stato contatto per il pick | `contact_state == 3` (**Portable Object**) |
| Distanza LEFT massima centri bbox | **40 pixel** |
| Pendenza LEFT | **`slope < -0.4`**, con `dx < 0` e `dy > 0` |
| Distanza FRONT massima oggetto-mano | **50 pixel** |
| `--pick-delay` | **2 frame** |
| Profondità: patch mediana | **7 × 7** pixel |
| Frazione minima di valori depth validi nella patch | **0.5** |
| Intervallo depth valido | **[0.15, 2.5] m** |
| Offset mano → oggetto in `base_link` | **(-0.02876, +0.01188, -0.06229) m** |
| Orientazione target | Quaternion **[0, 0, 1, 0]** |
| Gripper aperto / chiuso | **0.0 / 1.0** |
| Ultimo frame | Gripper **aperto (0)**; **nessun frame duplicato** |
| Grasp con bbox oggetto FRONT | **Nessun offset mano**, usa il punto oggetto 3D |
| Check salto waypoint consecutivi | Warning se > **0.08 m** nei due segmenti adiacenti al grasp |


## 4. ROI: coordinate e criterio di inclusione

Coordinate in pixel del frame originale **672 × 376**; origine immagine in alto a sinistra, `u` verso destra, `v` verso il basso.

### 4.1 FRONT: ROI rettangolare

```python
DEFAULT_ROIS = {
    'front': [93, 63, 648, 368],
    'left': [0, -1, 668, 333],   # fallback rettangolare, sostituito dal poligono LEFT
    'right': [120, 1, 665, 374], # presente nella configurazione ma non usato dal ciclo finale
}
```

FRONT: `93 <= u < 648` e `63 <= v < 368`.

### 4.2 LEFT: ROI poligonale

```python
ROI_POLYGONS = {
    'left': np.array([
        [65, 192],
        [315, 18],
        [610, 85],
        [470, 333],
        [105, 333],
    ], dtype=np.float32)
}
```

La funzione usa `cv2.pointPolygonTest(polygon, (u, v), False) >= 0`, includendo i punti sul bordo. Il poligono è stato disegnato per seguire approssimativamente l'area utile del tavolo, limitando falsi positivi su persona e joystick. **Le coordinate non sono normalizzate**: richiedono la risoluzione originale.

**Regola diversa tra categorie** (sia LEFT sia FRONT):

- **Mano:** conservare la detection se il **centro** `(cx,cy)` della bbox è dentro la ROI.
- **Oggetto:** conservare la detection solo se **tutti e quattro i vertici** `(x1,y1)`, `(x2,y1)`, `(x2,y2)`, `(x1,y2)` della bbox sono nella ROI.

I CSV originali in `work_dir/{left,front}/` non vengono sovrascritti; la funzione rigenera `work_dir/roi_filtered/{left,front}/`. Il report della selezione ROI è `roi_filtered/roi_filter_summary.json`.

## 5. Estrazione e rilevamento delle detection

1. `extract_images()` legge `data['traj']`, per `t=0..N-1`, prendendo `obs['camera_front_image']` e `obs['camera_lateral_left_image']`.
2. Salva PNG `frames/front/frame_00000.png`, `frames/left/frame_00000.png`, ecc.
3. Per la FRONT gli array immagine sono già **BGR** e vengono scritti senza ulteriore conversione; per la LEFT nel codice condiviso è usato `cv2.COLOR_RGB2BGR`. Immagini compresse sono decodificate con `cv2.imdecode`.
4. Chiama `hand_object_detector/demo.py` per **left** e **front** (con `--cuda`, `--net res101`, `--dataset pascal_voc`, `--thresh_hand 0.5`, `--thresh_obj 0.5`).
5. Usa i CSV raw `hand_detections.csv` e `object_detections.csv` di ogni camera e li passa al filtro ROI.

`--skip-detection` evita **estrazione immagini e inferenza detector**, ma non salta la rigenerazione dei CSV filtrati né le successive fasi di elaborazione. Richiede che tutti e quattro i CSV raw esistano già nella cartella di lavoro.

## 6. Identificazione del pick con la camera LEFT

Il modulo `filter_pick_matched_object.py` fornisce le funzioni `load()` e `inspect()`; **non viene eseguita la sua CLI a due camere**.

Per ciascun frame `t`:

1. `inspect(t, left_hands, left_objects, 0.5, 0.5, 40)` trova mano **destra** `side=1` con `contact_state=3`, mano e oggetto di score almeno 0.5, e valuta distanze euclidee fra centri bbox. Il modulo originario sceglie la coppia più vicina (con punteggi come criterio secondario).
2. `inspect_left_geometry(t, ...)` riesamina **tutte le coppie** mano destra portable/oggetto con gli stessi filtri preliminari, verifica distanza ≤40 pixel e vincoli geometrico-angolari.
3. Il nuovo frame è valido se `inspection['valid']` **e** `geometry['geometry_valid']` sono veri. Il primo frame valido cronologicamente è `detected_pick_frame = n`.
4. Il waypoint di grasp logico è `pick_frame = n + 2`.

Definizioni esatte nel riferimento pixel OpenCV:

```python
dx = object_cx - hand_cx
dy = object_cy - hand_cy
distance_px = np.hypot(dx, dy)
slope = dy / dx
valid_geometry = (
    dx < 0
    and dy > 0
    and distance_px <= 40
    and slope < -0.4
)
```

### File prodotti

- `pick_matched_timeline.csv`: valori per frame, esiti originali e nuove colonne di geometria (`geometry_dx`, `geometry_dy`, `geometry_slope`, `valid_after_geometry`, ecc.).
- `pick_matched_result.json`: primo pick trovato, camera LEFT, regola e soglia geometrica.

## 7. Ricostruzione della traiettoria 3D dalla FRONT

Ogni frame è descritto dalla **bbox della mano destra** nella camera FRONT; tra quelle disponibili con score ≥0.5 viene scelta la mano con score massimo. Il centro pixel `(u,v)` viene associato a `obs['camera_front_depth']`.

### 7.1 Calibrazione utilizzata

Dal modulo `utils/reconstruct_right_hand_3d.py`:

```python
W, H = 672, 376
FX = FY = 363.8398742675781
CX, CY = 337.5829162597656, 178.53465270996094

T_BASE_CAMERA = np.array([
    [-0.9995271385,  0.0017188839, -0.0307008924, 0.0258804482],
    [ 0.0210472614,  0.7661305314, -0.6423402694, 1.0325620478],
    [ 0.0224167827, -0.6426827011, -0.7658044356, 0.4595237268],
    [ 0.,            0.,            0.,           1.]
], dtype=np.float64)
```

**I numeri di calibrazione sono specifici per il setup.** Se cambia camera, calibrazione o montaggio, non riutilizzarli senza ricalibrarli.

### 7.2 Da RGB-D a `base_link`

- Patch depth mediana **7 × 7** attorno al centro bbox; sono validi solo i pixel finiti entro 0.15–2.5 m; almeno il 50% dei pixel della patch deve essere valido.
- Con depth `d`:

```python
Xc = (u - CX) * d / FX
Yc = (v - CY) * d / FY
Zc = d
p_base = (T_BASE_CAMERA @ [Xc, Yc, Zc, 1])[:3]
```

- Se manca una posizione della mano, la sequenza XYZ viene ricostruita con `np.interp` **per asse** sugli indici validi (compresi gli estremi della serie, estesi col valore più vicino). Se non c'è nessuna detection valida o ne esiste soltanto una, la conversione fallisce.
- La serie interpolata forma il percorso 3D iniziale; **solo il waypoint di grasp logico** verrà eventualmente sostituito.

## 8. Scelta finale del punto di grasp

Dopo il pick LEFT `n` e il pick logico `pick=n+2`, il builder tenta di trovare un oggetto nella FRONT usando i frame **dal pick verso n**, in ordine discendente: `pick, pick-1, ..., n`.

### 8.1 Priorità: centro oggetto FRONT

Per ogni candidato oggetto nella ROI FRONT:

1. score oggetto ≥0.5;
2. mano FRONT disponibile in quello stesso frame, con profondità valida;
3. distanza centri mano-oggetto **≤50 pixel**;
4. profondità mediana del centro oggetto valida e deproiezione 3D riuscita.

Fra i candidati validi si privilegia **il frame più recente** nella finestra; a parità di frame, l'oggetto **più vicino alla mano**. La stima ottenuta è il **centro 3D dell'oggetto** in `base_link`.

### 8.2 Fallback: mano FRONT + offset calibrato

Se non esistono candidati FRONT validi, viene usata la posizione 3D di una detection mano **direttamente valida**, scegliendo il frame più vicino al pick (preferendo quello precedente in caso di parità). Non viene usato il valore interpolato della mano per questa scelta.

```python
OFFSET_BASE = np.array([-0.01876, +0.00088, -0.06229])  # metri
p_grasp = p_hand_at_nearest_valid_frame + OFFSET_BASE
```

L'offset è espresso in **`base_link`**, non in pixel e non nel frame camera. Non viene applicato se è stato individuato un oggetto FRONT valido.

Sorgenti possibili nei report:

```text
front_hand_at_pick_offset
nearest_valid_front_hand_offset
```

### 8.3 Sostituzione del singolo waypoint

```python
hand_xyz_at_pick = positions[pick].copy()
positions[pick] = grasp_xyz
```

Gli altri `positions[t]` non vengono rimpiazzati. Il builder segnala se un passo prima/dopo il grasp è maggiore di 8 cm; la segnalazione **non blocca** di per sé il salvataggio.

## 9. Da traiettoria umana a PKL OSVI

Il loader originale si aspetta una `Trajectory` con `traj._data[t]` di 5 elementi:

```text
(obs, reward, done, info, action)
```

Il builder mantiene la struttura, gli altri campi di `obs`, i metadati e la lunghezza `N` della traiettoria; per ogni step:

```python
obs_new['eef_pos'] = positions[t]    # float64, XYZ base_link, (3,)
action = np.concatenate([
    positions[t],                    # 3 valori XYZ assoluti
    [0., 0., 1., 0.],                # quaternion fisso top-down, 4 valori
    [action_gripper[t]]              # gripper, 1 valore
])                                   # float64, shape (8,)
```


### Gripper

```python
action_gripper = np.full(N, 0.0)
action_gripper[pick:N-1] = 1.0
action_gripper[-1] = 0.0
```

Quindi:

- prima del pick: **0 = aperto**;
- dal pick al penultimo frame: **1 = chiuso**;
- ultimo frame originale: **0 = aperto**.


## 10. Diagnostica per traiettoria

In `front_grasp_final/task_XX/trajNNN/`:

| File/directory | Contenuto |
|---|---|
| `frames/front`, `frames/left` | Frame PNG estratti (quando non si usa `--skip-detection`) |
| `front/*.csv`, `left/*.csv` | Detection raw Faster R-CNN |
| `roi_filtered/front/*.csv`, `roi_filtered/left/*.csv` | Detection filtrate per ROI |
| `roi_filtered/roi_filter_summary.json` | Counts e descrizioni ROI |
| `pick_matched_timeline.csv` | Ispezione LEFT e geometria per frame |
| `pick_matched_result.json` | Primo pick rilevato |
| `grasp_candidates.csv` | Oggetti FRONT valutati, stato/reason, distanza, proiezione 3D |
| `grasp_front_source_overlay.png` | Croce del pixel **sorgente** sul frame FRONT scelto |
| `reconstruction_diagnostics.csv` | Mano FRONT: pixel, XYZ, stato depth, sorgente |
| `final_waypoints.csv` | XYZ mano e waypoint finali; `is_grasp` |
| `human_osvi_build_report.json` | Pick, origine grasp, offset, salti, frame interpolati, ecc. |

**Importante per la visualizzazione:** l'overlay `grasp_front_source_overlay.png` marca il **pixel sorgente** della bbox; nel fallback non riproietta il waypoint già corretto con l'offset. Per visualizzare l'offset effettivamente applicato occorre riproiettare la traiettoria del **PKL finale** con `plot_osvi_pkl_on_front.py`.

## 11. Verifica empirica dell'offset XYZ

Su 12 traiettorie del campione con detector FRONT oggetto valido, sono stati confrontati il punto 3D della mano e quello dell'oggetto **allo stesso `grasp_source_frame`**. Quattro traiettorie con fallback mano sono state escluse dalla stima (`task_01`, `task_05`, `task_08`, `task_13`).

**Mediane dell'offset `object_xyz - hand_xyz`:**

| Asse | Mediana | Media | Deviazione standard |
|---|---:|---:|---:|
| X | −2.88 cm | −2.96 cm | 1.12 cm |
| Y | +1.19 cm | +1.16 cm | 0.52 cm |
| Z | −6.23 cm | −6.27 cm | 1.33 cm |

**Validazione leave-one-out su 12 traiettorie:**

| Metodo | Errore medio | Errore mediano | Errore massimo |
|---|---:|---:|---:|
| Senza offset | 7.11 cm | 6.77 cm | 9.51 cm |
| Offset mediano globale | 1.70 cm | 1.67 cm | 2.76 cm |
| Mediana leave-one-out | 1.97 cm | 1.96 cm | 3.06 cm |

11/12 campioni hanno errore leave-one-out ≤3 cm, 12/12 ≤4 cm. Sono **errori rispetto alla bbox oggetto rilevata**, non una misura del successo di grasp del robot. Le detection oggetto non hanno verifica di identità automatica e il campione di validazione è piccolo.

## 12. Comandi per l'elaborazione completa

### 12.1 Attivazione

```bash
conda activate handobj_new
export ROOT="/home/mivia/Desktop/UR-Application/Federica/hand_tracking_test"
export PYTHONPATH="/home/mivia/Desktop/UR-Application/Federica/Multi-Task-LFD-Training-Framework/training/multi_task_il:/home/mivia/Desktop/UR-Application/Federica/Multi-Task-LFD-Training-Framework/training:${PYTHONPATH:-}"
cd "$ROOT"

find "$ROOT/dataset/human_reduced" -type f -name 'traj*.pkl' | wc -l
mkdir -p "$ROOT/front_grasp_final_logs"
```

### 12.2 Elaborazione di tutte le traiettorie

```bash
find "$ROOT/dataset/human_reduced" -type f -name 'traj*.pkl' -print0 | sort -z |
while IFS= read -r -d '' PKL; do
    TASK=$(basename "$(dirname "$PKL")")
    TRAJ=$(basename "$PKL" .pkl)

    OUTPUT="$ROOT/dataset/human_osvi_front_grasp_final/$TASK/$TRAJ.pkl"
    WORK="$ROOT/front_grasp_final/$TASK/$TRAJ"
    LOG="$ROOT/front_grasp_final_logs/${TASK}-${TRAJ}.log"

    echo "[PROCESSING] $TASK / $TRAJ"
    if [ -f "$OUTPUT" ] && [ -f "$WORK/human_osvi_build_report.json" ] &&
       grep -q '^\[DONE\] PKL' "$LOG" 2>/dev/null; then
        echo "[SKIPPED] $TASK / $TRAJ"
        continue
    fi

    mkdir -p "$(dirname "$OUTPUT")" "$WORK"
    DETECTION_ARGS=()
    if [ -f "$WORK/left/hand_detections.csv" ] &&
       [ -f "$WORK/left/object_detections.csv" ] &&
       [ -f "$WORK/front/hand_detections.csv" ] &&
       [ -f "$WORK/front/object_detections.csv" ]; then
        DETECTION_ARGS=(--skip-detection)
    fi

    python "$ROOT/utils/process_human_osvi_front_grasp.py" \
      --pkl "$PKL" \
      --output-pkl "$OUTPUT" \
      --work-dir "$WORK" \
      --repo "$ROOT/hand_object_detector" \
      --utils "$ROOT/utils" \
      --open-value 0.0 --closed-value 1.0 \
      --pick-delay 2 \
      --max-match-distance-px 40 \
      --left-max-slope=-0.4 \
      --front-object-max-distance-px 50 \
      --grasp-offset-xyz -0.02876 0.01188 -0.06229 \
      --depth-patch 7 \
      "${DETECTION_ARGS[@]}" --overwrite \
      > "$LOG" 2>&1

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] $TASK / $TRAJ"
    else
        echo "[FAILED] $TASK / $TRAJ"
        tail -n 12 "$LOG"
    fi
done
```

Questo ciclo evita di rieseguire il detector per le traiettorie con tutti i CSV raw già presenti nella **cartella `front_grasp_final`**. Le cartelle precedenti (`front_grasp_test`) non vengono riutilizzate implicitamente.

## 13. Plot della traiettoria finale (indipendente dalla pipeline)

```bash
python "$ROOT/utils/plot_osvi_pkl_on_front.py" \
  --pkl "$ROOT/dataset/human_osvi_front_grasp_final/task_01/traj001.pkl" \
  --output "$ROOT/front_grasp_final/task_01/traj001/traj_on_front_final.png" \
  --geometry-script "$ROOT/utils/reconstruct_right_hand_3d.py" \
  --pick-json "$ROOT/front_grasp_final/task_01/traj001/human_osvi_build_report.json" \
  --show-index --state osvi
```

- Lo script prende `obs['eef_pos']` da **tutti i frame** del PKL e li riproietta sul **primo frame FRONT**.
- Usa l'inversa `T_CAMERA_BASE = inv(T_BASE_CAMERA)` e le intrinseche di camera. I punti dietro la camera, non finiti o fuori frame non vengono disegnati.
- Traccia segmenti tra waypoint consecutivi visibili; marker per stato gripper, inizio e fine; con `--pick-json` marca il grasp (stella) e con `--show-index` mostra gli indici.






