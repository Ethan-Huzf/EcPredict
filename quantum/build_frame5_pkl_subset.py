#!/usr/bin/env python
"""Build a small reproduced frame pickle from dimer and V TSV files."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import MDAnalysis as mda
import numpy as np

try:
    from rdkit.Chem.rdmolfiles import MolFromXYZBlock

    RDKIT_AVAILABLE = True
except Exception:
    MolFromXYZBlock = None
    RDKIT_AVAILABLE = False


def atoms_to_xyz(atom_group) -> str:
    xyz_lines = [f"{len(atom_group)}", ""]
    for atom in atom_group:
        element = atom.element or atom.name[0]
        x, y, z = atom.position
        xyz_lines.append(f"{element} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(xyz_lines)


def unwrap_like_cudav9(atom_group):
    if RDKIT_AVAILABLE:
        mol = MolFromXYZBlock(atoms_to_xyz(atom_group))
        if mol is None:
            raise ValueError("RDKit failed to parse molecule from coordinates.")
        conf = mol.GetConformer()
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=float,
        )
    else:
        coords = np.round(np.asarray(atom_group.positions, dtype=float), 6)

    atom_group.positions = coords
    return atom_group


def shift_molecule_near(mol_to_shift, reference_mol, box: np.ndarray):
    ref_com = reference_mol.center_of_mass(pbc=False)
    shift_com = mol_to_shift.center_of_mass(pbc=False)
    shift_vector = np.round((shift_com - ref_com) / box[0:3]) * box[0:3]
    mol_to_shift.positions -= shift_vector
    return mol_to_shift


def compute_distance_row(universe, box: np.ndarray, res1: int, res2: int) -> np.ndarray:
    mol1_mda = universe.select_atoms(f"resid {res1}")
    mol2_mda = universe.select_atoms(f"resid {res2}")
    if len(mol1_mda) == 0 or len(mol2_mda) == 0:
        raise ValueError(f"Could not select resid pair {res1} {res2}.")

    mol1_unwrap = unwrap_like_cudav9(mol1_mda.atoms.copy())
    mol2_unwrap = unwrap_like_cudav9(mol2_mda.atoms.copy())
    mol2_shift = shift_molecule_near(mol2_unwrap, mol1_unwrap, box)

    coords = np.vstack((mol1_unwrap.positions, mol2_shift.positions))
    distance_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
    return distance_matrix.reshape(-1)


def read_dimer_rows(path: Path, start_index: int, end_index: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pair_index = int(row["pair_index"])
            if start_index <= pair_index < end_index:
                rows.append(row)
    rows.sort(key=lambda row: int(row["pair_index"]))
    return rows


def read_v_values(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pair_index = int(row["pair_index"])
            if row.get("status") != "ok":
                raise ValueError(f"pair_index {pair_index} does not have status=ok.")
            values[pair_index] = float(row["V_meV"])
    return values


def atom_list_for_pair(universe, res1: int, res2: int) -> list[str]:
    mol1 = universe.select_atoms(f"resid {res1}")
    mol2 = universe.select_atoms(f"resid {res2}")
    atom_symbols1 = [f"m1{atom.element or atom.name[0]}" for atom in mol1]
    atom_symbols2 = [f"m2{atom.element or atom.name[0]}" for atom in mol2]
    return atom_symbols1 + atom_symbols2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a subset dict_frame pkl.")
    parser.add_argument("--top", default="../box.prmtop")
    parser.add_argument("--traj", default="../production.nc")
    parser.add_argument("--frame", type=int, default=5)
    parser.add_argument("--dimers", default="results/frame5_dimers.tsv")
    parser.add_argument("--v-results", default="results/test_batch_v_00000_00010.tsv")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=10)
    parser.add_argument("--out", default="results/dict_frame5_first10_reproduced.pkl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dimer_rows = read_dimer_rows(Path(args.dimers), args.start_index, args.end_index)
    if not dimer_rows:
        raise ValueError("No dimer rows selected.")

    expected_indices = list(range(args.start_index, args.end_index))
    actual_indices = [int(row["pair_index"]) for row in dimer_rows]
    if actual_indices != expected_indices:
        raise ValueError(
            f"Selected pair_index values are not consecutive: {actual_indices[:5]}..."
        )

    v_values = read_v_values(Path(args.v_results))
    missing_v = [idx for idx in actual_indices if idx not in v_values]
    if missing_v:
        raise ValueError(f"Missing V values for pair_index values: {missing_v}")

    universe = mda.Universe(args.top, args.traj)
    ts = universe.trajectory[args.frame]
    box = ts.dimensions

    first = dimer_rows[0]
    atom_list = atom_list_for_pair(universe, int(first["res1"]), int(first["res2"]))

    rows = []
    for row in dimer_rows:
        pair_index = int(row["pair_index"])
        res1 = int(row["res1"])
        res2 = int(row["res2"])
        distance_row = compute_distance_row(universe, box, res1, res2)
        rows.append(np.concatenate([distance_row, [v_values[pair_index]]]))
        print(
            f"built pair_index={pair_index} res={res1}t{res2} "
            f"V_meV={v_values[pair_index]:.8g}",
            flush=True,
        )

    distance_matrix = np.vstack(rows).astype(np.float16)
    output_data = {
        "distance_matrix": distance_matrix,
        "atom_list": atom_list,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(output_data, handle)

    print(f"rdkit_available = {RDKIT_AVAILABLE}")
    print(f"shape = {distance_matrix.shape}")
    print(f"dtype = {distance_matrix.dtype}")
    print(f"out = {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
