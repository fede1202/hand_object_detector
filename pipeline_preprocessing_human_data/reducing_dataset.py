
#!/usr/bin/env python3

import argparse
import copy
import pickle
from pathlib import Path


def truncate_trajectory(traj, n_frames):
    """
    Crea una copia della Trajectory mantenendo
    soltanto i primi n_frames.
    """

    if isinstance(traj, list):
        return traj[:n_frames]

    if isinstance(traj, tuple):
        return traj[:n_frames]

    new_traj = copy.deepcopy(traj)
    original_len = len(traj)

    candidate_attributes = [
        "_data",
        "data",
        "_traj",
        "traj",
        "_trajectory",
        "trajectory",
        "steps",
        "_steps",
    ]

    for attr_name in candidate_attributes:

        if not hasattr(new_traj, attr_name):
            continue

        attr = getattr(new_traj, attr_name)

        try:
            if len(attr) != original_len:
                continue
        except TypeError:
            continue

        try:
            setattr(new_traj, attr_name, attr[:n_frames])
            return new_traj
        except Exception:
            continue

    raise RuntimeError(
        "Impossibile individuare la struttura interna "
        f"della Trajectory ({type(traj)}). "
        f"Attributi: {vars(traj) if hasattr(traj, '__dict__') else 'N/A'}"
    )


def reduce_pkl(input_path, output_path, last_frame):

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError("Il PKL non contiene un dizionario.")

    if "traj" not in data:
        raise KeyError("Chiave 'traj' non trovata nel PKL.")

    original_length = len(data["traj"])

    if original_length == 0:
        raise ValueError("La traiettoria originale è vuota.")

    if last_frame >= original_length:
        raise ValueError(
            f"Ultimo frame richiesto: {last_frame}. "
            f"La traiettoria contiene {original_length} frame "
            f"(indici 0..{original_length - 1})."
        )

    n_frames = last_frame + 1

    reduced_data = copy.deepcopy(data)

    reduced_data["traj"] = truncate_trajectory(
        data["traj"],
        n_frames,
    )

    reduced_length = len(reduced_data["traj"])

    if reduced_length != n_frames:
        raise RuntimeError(
            f"Lunghezza inattesa dopo il taglio: {reduced_length}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(reduced_data, f)

    print("\n" + "=" * 60)
    print("TRAJECTORY REDUCED SUCCESSFULLY")
    print("=" * 60)
    print(f"Input         : {input_path}")
    print(f"Output        : {output_path}")
    print(f"Original      : {original_length} frames")
    print(f"Reduced       : {reduced_length} frames")
    print(f"Frames kept   : 0 -> {last_frame}")
    print("=" * 60)


def main():

    parser = argparse.ArgumentParser(
        description="Riduce una singola traiettoria umana PKL."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Percorso del PKL originale.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Percorso del PKL ridotto.",
    )

    parser.add_argument(
        "--last-frame",
        required=True,
        type=int,
        help="Ultimo frame da mantenere, incluso.",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"PKL non trovato: {input_path}"
        )

    if input_path.suffix.lower() != ".pkl":
        raise ValueError("Il file input deve essere un PKL.")

    if input_path == output_path:
        raise ValueError(
            "Input e output non possono coincidere."
        )

    if args.last_frame < 0:
        raise ValueError(
            "--last-frame deve essere maggiore o uguale a zero."
        )

    reduce_pkl(
        input_path,
        output_path,
        args.last_frame,
    )


if __name__ == "__main__":
    main()
