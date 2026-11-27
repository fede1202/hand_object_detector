import argparse
import csv
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="hand_detections.csv")
parser.add_argument("--frames", required=True, help="hand_trajectory.csv")
parser.add_argument("--output", required=True)
parser.add_argument("--normalized", action="store_true")
args = parser.parse_args()

# Tutti i frame, anche quando la mano destra non viene rilevata
with open(args.frames, newline="") as f:
    frame_names = [row["frame"] for row in csv.DictReader(f)]

# Seleziona esclusivamente detection con side=1 (Right)
# In caso di più mani destre, mantiene quella con score più alto
right_hands = {}

with open(args.input, newline="") as f:
    for row in csv.DictReader(f):
        if int(float(row["side"])) != 1:
            continue

        frame = row["frame"]
        score = float(row["score"])

        if frame not in right_hands or score > float(right_hands[frame]["score"]):
            right_hands[frame] = row

# Trova il primo e l'ultimo contatto Right + Portable Object (classe 3)
contact_indices = []

for i, frame in enumerate(frame_names):
    detection = right_hands.get(frame)
    if detection is not None and int(float(detection["contact_state"])) == 3:
        contact_indices.append(i)

if not contact_indices:
    raise RuntimeError(
        "Nessun contatto Right + Portable Object rilevato. "
        "Impossibile determinare apertura e chiusura."
    )

first_contact = contact_indices[0]
last_contact = contact_indices[-1]

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)

x_key = "cx_norm" if args.normalized else "cx"
y_key = "cy_norm" if args.normalized else "cy"

with open(output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "x", "y", "gripper"])

    for i, frame in enumerate(frame_names):
        detection = right_hands.get(frame)

        if detection is None:
            x = math.nan
            y = math.nan
        else:
            x = float(detection[x_key])
            y = float(detection[y_key])

        # Chiuso dal primo all'ultimo contatto, estremi inclusi
        gripper = int(first_contact <= i <= last_contact)

        writer.writerow([frame, x, y, gripper])

print("=== RIGHT HAND TRAJECTORY ===")
print("Frame totali:", len(frame_names))
print("Frame con mano destra rilevata:", len(right_hands))
print("Primo contatto:", frame_names[first_contact])
print("Ultimo contatto:", frame_names[last_contact])
print("Gripper aperto dal frame successivo all'ultimo contatto")
print("Coordinate:", "normalizzate" if args.normalized else "pixel")
print("Salvato:", output)
