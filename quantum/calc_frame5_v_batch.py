#!/usr/bin/env python
"""Calculate electronic coupling V for a block of frame-5 dimers.

This script reuses the already validated single-pair implementation in
`calc_one_frame5_pair_v.py`, but keeps one trajectory load and one optional
phase-reference calculation for the whole block.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import MDAnalysis as mda
import numpy as np

from calc_one_frame5_pair_v import (
    HARTREE_TO_MEV,
    RDKIT_AVAILABLE,
    CalculationSettings,
    GridCache,
    calculate_v,
    choose_backend,
    get_homos,
    load_molecules,
    mda_to_pyscf,
    prepare_molecule,
    write_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate V values for a consecutive block of frame-5 dimers."
    )
    parser.add_argument("--top", default="../box.prmtop", help="AMBER prmtop path.")
    parser.add_argument("--traj", default="../production.nc", help="AMBER trajectory path.")
    parser.add_argument("--frame", type=int, default=5, help="Zero-based frame index.")
    parser.add_argument(
        "--pairs",
        default="results/frame5_dimers.tsv",
        help="Dimer list TSV with pair_index/res1/res2 columns.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        required=True,
        help="First pair_index to calculate, inclusive.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        required=True,
        help="Last pair_index boundary, exclusive.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output TSV path. Rows are written after each pair finishes.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to --out instead of overwriting it.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip pair_index values already present in --out with status=ok.",
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
        default=1,
        help="Residue used as the common phase reference. Use 0 to disable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load coordinates and write geometry rows; do not run DFT.",
    )
    return parser.parse_args()


def load_pair_rows(path: Path, start_index: int, end_index: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pair_index = int(row["pair_index"])
            if start_index <= pair_index < end_index:
                rows.append(row)
    return rows


def load_completed_pair_indices(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.exists() or path.stat().st_size == 0:
        return completed

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("status") == "ok" and row.get("pair_index"):
                completed.add(int(row["pair_index"]))
    return completed


def make_result_row(
    pair_row: dict[str, str],
    frame: int,
    backend_name: str,
    settings: CalculationSettings,
    phase_ref_resid: int | None,
) -> dict[str, object]:
    return {
        "pair_index": int(pair_row["pair_index"]),
        "frame": frame,
        "res1": int(pair_row["res1"]),
        "res2": int(pair_row["res2"]),
        "min_atom_distance_A": pair_row.get("min_atom_distance_A", ""),
        "com_distance_A": pair_row.get("com_distance_A", ""),
        "V_hartree": "nan",
        "V_meV": "nan",
        "homo1_idx": "",
        "homo2_idx": "",
        "homo1_label": "",
        "homo2_label": "",
        "min_overlap": "",
        "phase_ref_resid": phase_ref_resid if phase_ref_resid is not None else "",
        "backend": backend_name,
        "functional": settings.functional,
        "basis": settings.basis,
        "status": "started",
        "seconds": "",
        "error": "",
    }


def homo_label(homo_idx: int) -> str:
    return f"HOMO-{9 - homo_idx}" if homo_idx != 9 else "HOMO"


def main() -> int:
    args = parse_args()
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0.")
    if args.end_index <= args.start_index:
        raise ValueError("--end-index must be greater than --start-index.")

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
    pair_rows = load_pair_rows(Path(args.pairs), args.start_index, args.end_index)
    out_path = Path(args.out)
    completed = load_completed_pair_indices(out_path) if args.resume else set()
    phase_ref_resid = None if args.phase_ref_resid == 0 else args.phase_ref_resid

    print(f"frame = {args.frame}", flush=True)
    print(f"pair_index range = [{args.start_index}, {args.end_index})", flush=True)
    print(f"loaded_pair_rows = {len(pair_rows)}", flush=True)
    print(f"already_completed = {len(completed)}", flush=True)
    print(f"backend = {backend.name}", flush=True)
    print(f"rdkit_available = {RDKIT_AVAILABLE}", flush=True)
    print(f"functional = {settings.functional}", flush=True)
    print(f"basis = {settings.basis}", flush=True)
    print(f"out = {args.out}", flush=True)

    universe = mda.Universe(args.top, args.traj)
    ts = universe.trajectory[args.frame]
    box = ts.dimensions

    ref_mol = None
    ref_homo = None
    if phase_ref_resid is not None and not args.dry_run:
        ref_mda = prepare_molecule(universe.select_atoms(f"resid {phase_ref_resid}"))
        ref_mol = mda_to_pyscf(ref_mda, settings)
        ref_homo = get_homos(ref_mol, backend, settings)[:, 9]
        print(f"phase_ref_resid = {phase_ref_resid}", flush=True)
    elif args.dry_run:
        print("phase_ref_resid = skipped for dry run", flush=True)
    else:
        print("phase_ref_resid = none; using raw HOMO signs", flush=True)

    wrote_any = False
    n_ok = 0
    n_failed = 0
    n_skipped = 0
    total_start = time.time()

    for offset, pair_row in enumerate(pair_rows, start=1):
        pair_index = int(pair_row["pair_index"])
        if pair_index in completed:
            n_skipped += 1
            print(f"[{offset}/{len(pair_rows)}] pair_index={pair_index} skipped", flush=True)
            continue

        row = make_result_row(pair_row, args.frame, backend.name, settings, phase_ref_resid)
        pair_start = time.time()
        grid_cache = None
        print(
            f"[{offset}/{len(pair_rows)}] pair_index={pair_index} "
            f"res={row['res1']}t{row['res2']} start",
            flush=True,
        )

        try:
            mol1_mda, mol2_mda = load_molecules(
                universe, box, int(row["res1"]), int(row["res2"])
            )

            if args.dry_run:
                row["status"] = "dry_run"
            else:
                mol1 = mda_to_pyscf(mol1_mda, settings)
                mol2 = mda_to_pyscf(mol2_mda, settings)
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
                        "homo1_label": homo_label(homo_idx1),
                        "homo2_label": homo_label(homo_idx2),
                        "min_overlap": f"{min_overlap:.6f}",
                        "status": "ok",
                    }
                )
                n_ok += 1
                print(
                    f"[{offset}/{len(pair_rows)}] pair_index={pair_index} "
                    f"V_meV={v_mev:.8g} seconds={time.time() - pair_start:.2f}",
                    flush=True,
                )
        except Exception as exc:
            n_failed += 1
            row["status"] = "failed"
            row["error"] = repr(exc)
            traceback.print_exc(file=sys.stderr)
            print(
                f"[{offset}/{len(pair_rows)}] pair_index={pair_index} failed: {exc!r}",
                flush=True,
            )
        finally:
            if grid_cache is not None:
                grid_cache.clear()
            row["seconds"] = f"{time.time() - pair_start:.3f}"
            append_now = args.append or args.resume or wrote_any
            write_result(out_path, row, append=append_now)
            wrote_any = True

    print(
        "summary: "
        f"ok={n_ok} failed={n_failed} skipped={n_skipped} "
        f"elapsed_seconds={time.time() - total_start:.2f}",
        flush=True,
    )
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
