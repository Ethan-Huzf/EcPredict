#!/usr/bin/env python
"""Calculate electronic coupling V for one frame-5 dimer.

This is a clean, argument-driven version of the V calculation used by
`../glm5numericalv5.py`. It does not modify the original script.

The default execution path can run with plain PySCF on CPU. If `cupy` and
`gpu4pyscf` are available, `--backend auto` will use the GPU implementation.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from scipy.spatial import cKDTree

from pyscf import gto
from pyscf.dft import rks as cpu_rks

try:
    from rdkit.Chem.rdmolfiles import MolFromXYZBlock

    RDKIT_AVAILABLE = True
except Exception:
    MolFromXYZBlock = None
    RDKIT_AVAILABLE = False

try:
    import cupy as cupy_module
    from gpu4pyscf.dft import rks as gpu_rks

    GPU_AVAILABLE = True
    GPU_IMPORT_ERROR = None
except Exception as exc:
    cupy_module = None
    gpu_rks = None
    GPU_AVAILABLE = False
    GPU_IMPORT_ERROR = repr(exc)


HARTREE_TO_MEV = 27211.4


@dataclass
class Backend:
    name: str
    xp: object
    rks: object


@dataclass
class CalculationSettings:
    functional: str = "PW91"
    basis: str = "6-31g*"
    unit: str = "Angstrom"
    overlap_grid_level: int = 1
    dimer_grid_level: int = 2
    monomer_grid_level: int | None = None
    max_phase_insertions: int = 10
    overlap_threshold: float = 0.8


@dataclass
class PreparedMolecule:
    elements: list[str]
    positions: np.ndarray
    masses: np.ndarray

    def center_of_mass(self) -> np.ndarray:
        return np.average(self.positions, axis=0, weights=self.masses)


class GridCache:
    """Cache integration grids and AO values for repeated overlap checks."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.cache = {}

    def get_or_update(self, mol, grid_level: int):
        key = (id(mol), grid_level)
        if key not in self.cache:
            from pyscf.dft import gen_grid

            grids = gen_grid.Grids(mol)
            grids.level = grid_level
            grids.build()
            coords_np = np.asarray(grids.coords)
            weights_np = np.asarray(grids.weights)
            ao_values_np = np.asarray(mol.eval_gto("GTOval", coords_np))
            self.cache[key] = (
                self.backend.xp.asarray(coords_np),
                self.backend.xp.asarray(weights_np),
                self.backend.xp.asarray(ao_values_np),
            )
        return self.cache[key]

    def clear(self) -> None:
        self.cache.clear()
        if self.backend.name == "gpu" and cupy_module is not None:
            cupy_module.get_default_memory_pool().free_all_blocks()


def choose_backend(name: str) -> Backend:
    if name == "gpu":
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "Requested --backend gpu, but cupy/gpu4pyscf is unavailable. "
                f"Import error: {GPU_IMPORT_ERROR}"
            )
        return Backend(name="gpu", xp=cupy_module, rks=gpu_rks)

    if name == "auto" and GPU_AVAILABLE:
        return Backend(name="gpu", xp=cupy_module, rks=gpu_rks)

    return Backend(name="cpu", xp=np, rks=cpu_rks)


def to_numpy(value) -> np.ndarray:
    if cupy_module is not None and isinstance(value, cupy_module.ndarray):
        return cupy_module.asnumpy(value)
    return np.asarray(value)


def to_float(value) -> float:
    return float(np.asarray(to_numpy(value)))


def kabsch(coord_var: np.ndarray, coord_ref: np.ndarray) -> np.ndarray:
    covar = np.dot(coord_var.T, coord_ref)
    v, _s, wt = np.linalg.svd(covar)
    if (np.linalg.det(v) * np.linalg.det(wt)) < 0.0:
        v[:, -1] *= -1.0
    return np.dot(v, wt)


def centroid(coords: np.ndarray) -> np.ndarray:
    return coords.mean(axis=0)


def atoms_to_xyz(atom_group) -> str:
    xyz_lines = [f"{len(atom_group)}", ""]
    for atom in atom_group:
        element = atom.element or atom.name[0]
        x, y, z = atom.position
        xyz_lines.append(f"{element} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(xyz_lines)


def prepare_molecule(atom_group) -> PreparedMolecule:
    """Return independent molecule coordinates using cudav9-style precision.

    The original helper writes an XYZ string with six decimal places, then asks
    RDKit to read it back. In environments without RDKit, rounding the coordinate
    copy to six decimals is the same coordinate operation for this script's use.
    """
    elements = [atom.element or atom.name[0] for atom in atom_group]
    masses = np.asarray(atom_group.masses, dtype=float)

    if RDKIT_AVAILABLE:
        mol = MolFromXYZBlock(atoms_to_xyz(atom_group))
        if mol is None:
            raise ValueError("RDKit failed to parse molecule from coordinates.")
        conf = mol.GetConformer()
        positions = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=float,
        )
    else:
        positions = np.round(np.asarray(atom_group.positions, dtype=float), 6)

    return PreparedMolecule(elements=elements, positions=positions, masses=masses)


def shift_molecule_near(mol_to_shift: PreparedMolecule, reference_mol: PreparedMolecule, box: np.ndarray):
    shift_vector = (
        np.round((mol_to_shift.center_of_mass() - reference_mol.center_of_mass()) / box[0:3])
        * box[0:3]
    )
    mol_to_shift.positions -= shift_vector
    return mol_to_shift


def mda_to_pyscf(molecule: PreparedMolecule, settings: CalculationSettings):
    atoms = []
    for element, position in zip(molecule.elements, molecule.positions):
        x, y, z = position
        atoms.append(f"{element} {x:.6f} {y:.6f} {z:.6f}")
    return gto.M(atom="; ".join(atoms), basis=settings.basis, unit=settings.unit, verbose=0)


def pyscf_mol_from_bohr_coords(template_mol, coords_bohr: np.ndarray, settings: CalculationSettings):
    atoms = []
    for idx, pos in enumerate(coords_bohr):
        symbol = template_mol.atom_symbol(idx)
        atoms.append(f"{symbol} {pos[0]:.16f} {pos[1]:.16f} {pos[2]:.16f}")
    return gto.M(atom="; ".join(atoms), basis=settings.basis, unit="Bohr", verbose=0)


def get_transformation_from_ref_to_current(ref_mol, curr_mol) -> tuple[np.ndarray, np.ndarray]:
    coord_ref = np.asarray(ref_mol.atom_coords())
    coord_curr = np.asarray(curr_mol.atom_coords())
    centroid_ref = centroid(coord_ref)
    centroid_curr = centroid(coord_curr)
    coord_ref_cen = coord_ref - centroid_ref
    coord_curr_cen = coord_curr - centroid_curr
    rotation = kabsch(coord_ref_cen, coord_curr_cen)
    translation = centroid_curr - np.dot(centroid_ref, rotation)
    return rotation, translation


def evaluate_orbital_on_grid_fast(mol, orbital_coeff, grid_cache: GridCache, grid_level: int):
    coords, weights, ao_values = grid_cache.get_or_update(mol, grid_level)
    orbital_values = grid_cache.backend.xp.dot(ao_values, orbital_coeff)
    return coords, orbital_values, weights


def evaluate_all_orbitals_on_grid_fast(mol, orbital_coeffs, grid_cache: GridCache, grid_level: int):
    coords, weights, ao_values = grid_cache.get_or_update(mol, grid_level)
    orbital_values = grid_cache.backend.xp.dot(ao_values, orbital_coeffs)
    return coords, orbital_values, weights


def interpolate_orbital_kdtree(backend: Backend, coords_source, values_source, coords_target, k: int = 3):
    coords_source_np = to_numpy(coords_source)
    values_source_np = to_numpy(values_source)
    coords_target_np = to_numpy(coords_target)

    tree = cKDTree(coords_source_np)
    distances, indices = tree.query(coords_target_np, k=k)

    if k == 1:
        values_target_np = values_source_np[indices]
    else:
        distances = np.maximum(distances, 1e-10)
        weights = 1.0 / distances
        weights = weights / weights.sum(axis=1, keepdims=True)
        values_target_np = np.sum(values_source_np[indices] * weights, axis=1)

    return backend.xp.asarray(values_target_np)


def interpolate_all_orbitals_kdtree(
    backend: Backend, coords_source, values_source_all, coords_target, k: int = 3
):
    coords_source_np = to_numpy(coords_source)
    values_source_all_np = to_numpy(values_source_all)
    coords_target_np = to_numpy(coords_target)

    tree = cKDTree(coords_source_np)
    distances, indices = tree.query(coords_target_np, k=k)

    if k == 1:
        values_target_all_np = values_source_all_np[indices]
    else:
        distances = np.maximum(distances, 1e-10)
        weights = 1.0 / distances
        weights = weights / weights.sum(axis=1, keepdims=True)
        values_target_all_np = np.sum(
            values_source_all_np[indices] * weights[:, :, np.newaxis],
            axis=1,
        )

    return backend.xp.asarray(values_target_all_np)


def calculate_numerical_overlap_batch(
    ref_mol,
    ref_orbitals,
    mol_a,
    orbitals_a,
    grid_cache: GridCache,
    grid_level: int,
):
    xp = grid_cache.backend.xp
    coords_ref, values_ref_all, _weights_ref = evaluate_all_orbitals_on_grid_fast(
        ref_mol, ref_orbitals, grid_cache, grid_level
    )
    rotation, translation = get_transformation_from_ref_to_current(ref_mol, mol_a)
    coords_ref_aligned = xp.dot(coords_ref, xp.asarray(rotation)) + xp.asarray(translation)

    coords_a, values_a_all, weights_a = evaluate_all_orbitals_on_grid_fast(
        mol_a, orbitals_a, grid_cache, grid_level
    )
    values_ref_on_a_grid_all = interpolate_all_orbitals_kdtree(
        grid_cache.backend, coords_ref_aligned, values_ref_all, coords_a
    )
    return xp.dot((values_ref_on_a_grid_all * weights_a[:, np.newaxis]).T, values_a_all)


def calculate_numerical_overlap_v3(
    ref_mol,
    ref_homo,
    mol_a,
    homo_a,
    grid_cache: GridCache,
    grid_level: int,
):
    xp = grid_cache.backend.xp
    coords_ref, values_ref, _weights_ref = evaluate_orbital_on_grid_fast(
        ref_mol, ref_homo, grid_cache, grid_level
    )
    rotation, translation = get_transformation_from_ref_to_current(ref_mol, mol_a)
    coords_ref_aligned = xp.dot(coords_ref, xp.asarray(rotation)) + xp.asarray(translation)

    coords_a, values_a, weights_a = evaluate_orbital_on_grid_fast(
        mol_a, homo_a, grid_cache, grid_level
    )
    values_ref_on_a_grid = interpolate_orbital_kdtree(
        grid_cache.backend, coords_ref_aligned, values_ref, coords_a
    )
    return xp.sum(values_ref_on_a_grid * values_a * weights_a)


def get_homos(mol, backend: Backend, settings: CalculationSettings):
    mf = backend.rks.RKS(mol).density_fit()
    mf.xc = settings.functional
    mf.with_df.auxbasis = "weigend"
    mf.grids.prune = True
    if settings.monomer_grid_level is not None:
        mf.grids.level = settings.monomer_grid_level
    mf.kernel()
    mo_coeff = backend.xp.asarray(mf.mo_coeff)
    homo_idx = mol.nelectron // 2 - 1
    return mo_coeff[:, homo_idx - 9 : homo_idx + 1]


def make_transition_structure(mol, mol_ref, fraction: float, settings: CalculationSettings):
    coord_var = np.asarray(mol.atom_coords())
    coord_ref = np.asarray(mol_ref.atom_coords())
    trans = centroid(coord_ref)
    coord_var_cen = coord_var - centroid(coord_var)
    coord_ref_cen = coord_ref - centroid(coord_ref)
    rotation = kabsch(coord_var_cen, coord_ref_cen)
    coord_var_shifted = np.dot(coord_var_cen, rotation) + trans
    average_coords = (1.0 - fraction) * coord_var_shifted + fraction * coord_ref
    return pyscf_mol_from_bohr_coords(mol, average_coords, settings)


def find_best_homo_numerical_v3(
    homos_a,
    mol_a,
    ref_homo,
    ref_mol,
    grid_cache: GridCache,
    settings: CalculationSettings,
):
    ref_homo_2d = ref_homo.reshape(-1, 1)
    overlap_matrix = calculate_numerical_overlap_batch(
        ref_mol,
        ref_homo_2d,
        mol_a,
        homos_a,
        grid_cache,
        settings.overlap_grid_level,
    )
    overlap_values = overlap_matrix.flatten()
    best_idx = int(to_float(grid_cache.backend.xp.argmax(grid_cache.backend.xp.abs(overlap_values))))
    max_overlap = overlap_values[best_idx]
    best_homo = homos_a[:, best_idx]
    return best_homo, best_idx, max_overlap


def get_correct_homo_fast(
    homos,
    curr_mol,
    ref_homo,
    ref_mol,
    grid_cache: GridCache,
    settings: CalculationSettings,
):
    homo, best_idx, max_overlap = find_best_homo_numerical_v3(
        homos, curr_mol, ref_homo, ref_mol, grid_cache, settings
    )
    if abs(to_float(max_overlap)) >= settings.overlap_threshold:
        return homo, best_idx, max_overlap, np.sign(to_float(max_overlap))

    print(
        f"[LOW OVERLAP] initial overlap {abs(to_float(max_overlap)):.4f} "
        f"< {settings.overlap_threshold}",
        flush=True,
    )

    structures = [ref_mol, curr_mol]
    orbitals = [ref_homo, homo]
    homos_list = [None, homos]
    insertion_count = 0
    idx = 0
    prev_overlap_val = 1.0

    while idx < len(structures) - 1:
        overlap = calculate_numerical_overlap_v3(
            structures[idx],
            orbitals[idx],
            structures[idx + 1],
            orbitals[idx + 1],
            grid_cache,
            settings.overlap_grid_level,
        )

        if abs(to_float(overlap)) < settings.overlap_threshold:
            insertion_count += 1
            if insertion_count > settings.max_phase_insertions:
                print("[LOW OVERLAP] reached maximum structure insertions", flush=True)
                break

            overlap_prev = abs(to_float(prev_overlap_val)) if idx > 0 else 1.0
            overlap_next = abs(to_float(overlap))
            fraction = overlap_next / (overlap_prev + overlap_next)

            mol_mid = make_transition_structure(
                structures[idx], structures[idx + 1], fraction, settings
            )
            structures.insert(idx + 1, mol_mid)
            homos_mid = get_homos(mol_mid, grid_cache.backend, settings)
            homos_list.insert(idx + 1, homos_mid)

            best_mid_homo, _mid_idx, _mid_overlap = find_best_homo_numerical_v3(
                homos_mid,
                mol_mid,
                orbitals[idx],
                structures[idx],
                grid_cache,
                settings,
            )
            orbitals.insert(idx + 1, best_mid_homo)

            next_idx = idx + 2
            if next_idx < len(structures):
                best_next_homo, _next_idx, _next_overlap = find_best_homo_numerical_v3(
                    homos_list[next_idx],
                    structures[next_idx],
                    best_mid_homo,
                    mol_mid,
                    grid_cache,
                    settings,
                )
                orbitals[next_idx] = best_next_homo
        else:
            prev_overlap_val = overlap
            idx += 1

    overlap_list = [
        calculate_numerical_overlap_v3(
            structures[i],
            orbitals[i],
            structures[i + 1],
            orbitals[i + 1],
            grid_cache,
            settings.overlap_grid_level,
        )
        for i in range(len(structures) - 1)
    ]
    phase = np.sign(math.prod(to_float(overlap) for overlap in overlap_list))
    return homo, best_idx, max_overlap, phase


def calculate_v(
    mol1,
    mol2,
    backend: Backend,
    grid_cache: GridCache,
    settings: CalculationSettings,
    ref_mol=None,
    ref_homo=None,
    ref_idx: int = 9,
):
    xp = backend.xp
    homos1 = get_homos(mol1, backend, settings)
    homos2 = get_homos(mol2, backend, settings)

    if ref_mol is None or ref_homo is None:
        homo_idx1 = 9
        homo_idx2 = 9
        homo1 = homos1[:, homo_idx1]
        homo2 = homos2[:, homo_idx2]
        max_overlap1 = 1.0
        max_overlap2 = 1.0
    else:
        homo1, homo_idx1, max_overlap1, phase1 = get_correct_homo_fast(
            homos1, mol1, ref_homo, ref_mol, grid_cache, settings
        )
        if abs(float(phase1)) < 1e-5:
            return np.nan, homo_idx1, 100, 0.0
        homo1 = homo1 * float(np.sign(phase1))

        homo2, homo_idx2, max_overlap2, phase2 = get_correct_homo_fast(
            homos2, mol2, ref_homo, ref_mol, grid_cache, settings
        )
        if abs(float(phase2)) < 1e-5:
            return np.nan, 100, homo_idx2, 0.0
        homo2 = homo2 * float(np.sign(phase2))

        if homo_idx1 != ref_idx:
            print(f"Base 1 hole hopping from HOMO-{9 - ref_idx} to HOMO-{9 - homo_idx1}")
        if homo_idx2 != ref_idx:
            print(f"Base 2 hole hopping from HOMO-{9 - ref_idx} to HOMO-{9 - homo_idx2}")

    dimer = gto.M(
        atom=mol1.atom + ";" + mol2.atom,
        basis=settings.basis,
        unit=settings.unit,
        verbose=0,
    )
    mf_dimer = backend.rks.RKS(dimer).density_fit()
    mf_dimer.grids.level = settings.dimer_grid_level
    mf_dimer.xc = settings.functional
    mf_dimer.kernel()

    fock = xp.asarray(mf_dimer.get_fock())
    overlap_matrix = xp.asarray(dimer.intor("int1e_ovlp"))

    nao1 = mol1.nao_nr()
    nao2 = mol2.nao_nr()
    nao_total = dimer.nao_nr()
    psi1 = xp.zeros(nao_total)
    psi2 = xp.zeros(nao_total)
    psi1[:nao1] = homo1
    psi2[nao1 : nao1 + nao2] = homo2

    s12 = xp.dot(psi1, xp.matmul(overlap_matrix, psi2))
    e1 = xp.dot(psi1, xp.matmul(fock, psi1))
    e2 = xp.dot(psi2, xp.matmul(fock, psi2))
    j12 = xp.dot(psi1, xp.matmul(fock, psi2))

    denominator = 1 - s12 * s12
    if abs(to_float(denominator)) < 1e-10:
        warnings.warn(f"Very small denominator in J12_eff calculation: {to_float(denominator):.3e}")
        j12_eff = xp.asarray(np.nan)
    else:
        j12_eff = (j12 - 0.5 * (e1 + e2) * s12) / denominator

    min_overlap = min(abs(to_float(max_overlap1)), abs(to_float(max_overlap2)))
    return to_float(j12_eff), homo_idx1, homo_idx2, min_overlap


def load_pair_from_tsv(path: Path, pair_index: int) -> tuple[int, int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["pair_index"]) == pair_index:
                return int(row["res1"]), int(row["res2"]), row
    raise ValueError(f"pair_index {pair_index} not found in {path}")


def load_molecules(universe, box: np.ndarray, res1: int, res2: int):
    mol1_mda = universe.select_atoms(f"resid {res1}")
    mol2_mda = universe.select_atoms(f"resid {res2}")
    if len(mol1_mda) == 0 or len(mol2_mda) == 0:
        raise ValueError(f"Could not select resid pair {res1} {res2}.")

    mol1_unwrap = prepare_molecule(mol1_mda)
    mol2_unwrap = prepare_molecule(mol2_mda)
    mol2_shift = shift_molecule_near(mol2_unwrap, mol1_unwrap, box)
    return mol1_unwrap, mol2_shift


def geometry_summary(mol1, mol2, universe, box: np.ndarray, res1: int, res2: int) -> tuple[float, float]:
    cross_distances = np.linalg.norm(
        mol1.positions[:, np.newaxis, :] - mol2.positions[np.newaxis, :, :],
        axis=2,
    )
    all_coords = universe.atoms.positions.reshape(len(universe.residues), -1, 3)
    centers = all_coords.mean(axis=1)
    from MDAnalysis.lib.distances import distance_array

    com_distance = distance_array(centers[[res1 - 1]], centers[[res2 - 1]], box=box)[0, 0]
    return float(cross_distances.min()), float(com_distance)


def write_result(path: Path, row: dict[str, object], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_index",
        "frame",
        "res1",
        "res2",
        "min_atom_distance_A",
        "com_distance_A",
        "V_hartree",
        "V_meV",
        "homo1_idx",
        "homo2_idx",
        "homo1_label",
        "homo2_label",
        "min_overlap",
        "phase_ref_resid",
        "backend",
        "functional",
        "basis",
        "status",
        "seconds",
        "error",
    ]
    needs_header = not append or not path.exists() or path.stat().st_size == 0
    with path.open("a" if append else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate V for one selected dimer.")
    parser.add_argument("--top", default="../box.prmtop", help="AMBER prmtop path.")
    parser.add_argument("--traj", default="../production.nc", help="AMBER trajectory path.")
    parser.add_argument("--frame", type=int, default=5, help="Zero-based frame index.")
    parser.add_argument("--res1", type=int, help="First residue id.")
    parser.add_argument("--res2", type=int, help="Second residue id.")
    parser.add_argument(
        "--pair-index",
        type=int,
        help="Read res1/res2 from --pairs using this pair_index.",
    )
    parser.add_argument(
        "--pairs",
        default="results/frame5_dimers.tsv",
        help="Dimer list TSV used with --pair-index.",
    )
    parser.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--functional", default="PW91")
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--overlap-grid-level", type=int, default=1)
    parser.add_argument("--dimer-grid-level", type=int, default=2)
    parser.add_argument("--monomer-grid-level", type=int)
    parser.add_argument("--overlap-threshold", type=float, default=0.8)
    parser.add_argument("--max-phase-insertions", type=int, default=10)
    parser.add_argument(
        "--phase-ref-resid",
        type=int,
        help="Use this residue as a common phase reference. Omit for raw HOMO signs.",
    )
    parser.add_argument("--out", help="Optional TSV file for the one-row result.")
    parser.add_argument("--append", action="store_true", help="Append to --out.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load coordinates and print geometry; do not run DFT.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pair_index is not None:
        res1, res2, pair_row = load_pair_from_tsv(Path(args.pairs), args.pair_index)
    else:
        if args.res1 is None or args.res2 is None:
            raise ValueError("Provide either --pair-index or both --res1 and --res2.")
        res1, res2 = args.res1, args.res2
        pair_row = {}

    settings = CalculationSettings(
        functional=args.functional,
        basis=args.basis,
        overlap_grid_level=args.overlap_grid_level,
        dimer_grid_level=args.dimer_grid_level,
        monomer_grid_level=args.monomer_grid_level,
        max_phase_insertions=args.max_phase_insertions,
        overlap_threshold=args.overlap_threshold,
    )
    backend = choose_backend(args.backend)

    start_time = time.time()
    row: dict[str, object] = {
        "pair_index": args.pair_index if args.pair_index is not None else "",
        "frame": args.frame,
        "res1": res1,
        "res2": res2,
        "min_atom_distance_A": "",
        "com_distance_A": "",
        "V_hartree": "nan",
        "V_meV": "nan",
        "homo1_idx": "",
        "homo2_idx": "",
        "homo1_label": "",
        "homo2_label": "",
        "min_overlap": "",
        "phase_ref_resid": args.phase_ref_resid if args.phase_ref_resid is not None else "",
        "backend": backend.name,
        "functional": settings.functional,
        "basis": settings.basis,
        "status": "started",
        "seconds": "",
        "error": "",
    }

    try:
        universe = mda.Universe(args.top, args.traj)
        ts = universe.trajectory[args.frame]
        box = ts.dimensions
        mol1_mda, mol2_mda = load_molecules(universe, box, res1, res2)
        min_distance, com_distance = geometry_summary(
            mol1_mda, mol2_mda, universe, box, res1, res2
        )
        row["min_atom_distance_A"] = f"{min_distance:.6f}"
        row["com_distance_A"] = f"{com_distance:.6f}"

        print(f"frame = {args.frame}")
        print(f"res1 res2 = {res1} {res2}")
        print(f"backend = {backend.name}")
        print(f"rdkit_available = {RDKIT_AVAILABLE}")
        print(f"functional = {settings.functional}")
        print(f"basis = {settings.basis}")
        print(f"min_atom_distance_A = {min_distance:.6f}")
        print(f"com_distance_A = {com_distance:.6f}")

        if args.dry_run:
            row["status"] = "dry_run"
            return_code = 0
        else:
            mol1 = mda_to_pyscf(mol1_mda, settings)
            mol2 = mda_to_pyscf(mol2_mda, settings)

            ref_mol = None
            ref_homo = None
            if args.phase_ref_resid is not None:
                ref_mda = prepare_molecule(
                    universe.select_atoms(f"resid {args.phase_ref_resid}")
                )
                ref_mol = mda_to_pyscf(ref_mda, settings)
                ref_homo = get_homos(ref_mol, backend, settings)[:, 9]
                print(f"phase_ref_resid = {args.phase_ref_resid}", flush=True)
            else:
                print("phase_ref_resid = none; using raw HOMO signs", flush=True)

            grid_cache = GridCache(backend)
            v_hartree, homo_idx1, homo_idx2, min_overlap = calculate_v(
                mol1,
                mol2,
                backend,
                grid_cache,
                settings,
                ref_mol=ref_mol,
                ref_homo=ref_homo,
            )
            v_mev = v_hartree * HARTREE_TO_MEV

            row.update(
                {
                    "V_hartree": f"{v_hartree:.12g}",
                    "V_meV": f"{v_mev:.8g}",
                    "homo1_idx": homo_idx1,
                    "homo2_idx": homo_idx2,
                    "homo1_label": f"HOMO-{9 - homo_idx1}" if homo_idx1 != 9 else "HOMO",
                    "homo2_label": f"HOMO-{9 - homo_idx2}" if homo_idx2 != 9 else "HOMO",
                    "min_overlap": f"{min_overlap:.6f}",
                    "status": "ok",
                }
            )
            print(f"V_hartree = {v_hartree:.12g}")
            print(f"V_meV = {v_mev:.8g}")
            print(f"homo1_idx homo2_idx = {homo_idx1} {homo_idx2}")
            print(f"min_overlap = {min_overlap:.6f}")
            return_code = 0

    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
        traceback.print_exc(file=sys.stderr)
        return_code = 1
    finally:
        row["seconds"] = f"{time.time() - start_time:.3f}"
        if args.out:
            write_result(Path(args.out), row, append=args.append)
            print(f"out = {args.out}")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
