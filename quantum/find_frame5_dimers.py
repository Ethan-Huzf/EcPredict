#!/usr/bin/env python
"""Find dimer candidates for one trajectory frame.

This script reproduces the dimer selection rule from ../splitfilmv5(1).py:

1. Treat each residue as one molecule.
2. Pre-filter by geometric-center distance:
   center_distance <= molecule_diameter + cutoff
3. Keep a pair when the minimum periodic atom-atom distance is <= cutoff.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array, self_distance_array


def get_box_vectors(box: np.ndarray) -> np.ndarray:
    """Convert AMBER box dimensions to 3 box vectors."""
    lx, ly, lz, alpha, beta, gamma = box
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    a = np.array([lx, 0.0, 0.0])
    b = np.array([ly * np.cos(gamma), ly * np.sin(gamma), 0.0])
    c = np.array(
        [
            lz * np.cos(beta),
            lz * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
            lz
            * np.sqrt(
                1
                - np.cos(beta) ** 2
                - (
                    (np.cos(alpha) - np.cos(beta) * np.cos(gamma))
                    / np.sin(gamma)
                )
                ** 2
            ),
        ]
    )
    return np.asarray([a, b, c])


def small_molecule_distances(
    pos1: np.ndarray, pos2: np.ndarray, box_vectors: np.ndarray
) -> np.ndarray:
    """Periodic atom-atom distances between two small molecules."""
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    box = np.asarray(box_vectors)

    if box.ndim == 1:
        box = np.diag(box)

    inv_box = np.linalg.inv(box.T)
    frac1 = pos1 @ inv_box
    frac2 = pos2 @ inv_box
    delta_frac = frac1[:, np.newaxis, :] - frac2[np.newaxis, :, :]
    delta_frac -= np.round(delta_frac)
    delta_cart = np.einsum("...ij,jk->...ik", delta_frac, box.T)
    return np.linalg.norm(delta_cart, axis=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find dimer candidates for a selected frame."
    )
    parser.add_argument("--top", default="../box.prmtop", help="AMBER prmtop path.")
    parser.add_argument(
        "--traj", default="../production.nc", help="AMBER NetCDF trajectory path."
    )
    parser.add_argument("--frame", type=int, default=5, help="Zero-based frame index.")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Atom-atom distance cutoff in Angstrom.",
    )
    parser.add_argument(
        "--out",
        default="results/frame5_dimers.tsv",
        help="Output TSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    universe = mda.Universe(args.top, args.traj)
    ts = universe.trajectory[args.frame]
    box = ts.dimensions
    box_vectors = get_box_vectors(box)

    num_residues = len(universe.residues)
    atoms_per_residue = universe.residues[0].atoms.n_atoms
    all_coords = universe.atoms.positions.reshape(num_residues, atoms_per_residue, 3)

    # Keep the original script's geometric center convention, not mass COM.
    mol_centers = all_coords.mean(axis=1)
    com_distances = distance_array(mol_centers, mol_centers, box=box)

    first_molecule = universe.select_atoms("resid 1")
    molecule_diameter = self_distance_array(first_molecule.positions, box=box).max()
    com_cutoff = molecule_diameter + args.cutoff

    records: list[tuple[int, int, int, float, float]] = []
    com_candidate_count = 0
    pair_index = 0

    for i in range(num_residues):
        candidate_js = np.where(com_distances[i, i + 1 :] <= com_cutoff)[0] + i + 1
        com_candidate_count += len(candidate_js)

        for j in candidate_js:
            min_atom_distance = small_molecule_distances(
                all_coords[i], all_coords[j], box_vectors
            ).min()
            if min_atom_distance <= args.cutoff:
                records.append(
                    (
                        pair_index,
                        args.frame,
                        i + 1,
                        j + 1,
                        float(min_atom_distance),
                        float(com_distances[i, j]),
                    )
                )
                pair_index += 1

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "pair_index\tframe\tres1\tres2\tmin_atom_distance_A\tcom_distance_A\n"
        )
        for record in records:
            handle.write(
                f"{record[0]}\t{record[1]}\t{record[2]}\t{record[3]}"
                f"\t{record[4]:.6f}\t{record[5]:.6f}\n"
            )

    print(f"frame = {args.frame}")
    print(f"num_residues = {num_residues}")
    print(f"atoms_per_residue = {atoms_per_residue}")
    print(f"molecule_diameter_A = {molecule_diameter:.6f}")
    print(f"com_cutoff_A = {com_cutoff:.6f}")
    print(f"com_candidates = {com_candidate_count}")
    print(f"dimer_pairs = {len(records)}")
    print(f"output = {out_path}")


if __name__ == "__main__":
    main()
