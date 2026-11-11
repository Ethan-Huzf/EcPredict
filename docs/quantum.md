# Optional geometry and coupling-reproduction utilities

The four scripts under `quantum/` are byte-identical copies of the argument-driven Frame5 reproduction utilities. They accept user-provided AMBER topology/trajectory and pair-list paths. They are not a bundled MD dataset or a validated end-to-end reconstruction of every archived coupling label.

## Dimer selection

`find_frame5_dimers.py` first screens molecular geometric centers and then checks the minimum periodic atom–atom separation. Residues are treated as pentacene molecules, and H atoms are included. The default contact cutoff is 5.0 Å. The atom count/order and periodic-image handling must match the input film.

`--frame` is explicitly a **zero-based Python trajectory index**. The names Frame5, Frame10000, Frame20000 and Frame30000 are archived dataset labels; do not infer one-based numbering or elapsed time from those names alone. Exact reconstruction of an archive additionally requires the original pair membership/order and extraction convention. Newly finding candidates can produce a different pair list from the archived, filtered dataset.

## Coupling calculation

`calc_one_frame5_pair_v.py` implements

```text
V12 = [H12 - 0.5*(epsilon1 + epsilon2)*S12] / (1 - S12**2)
H12 = c1.T @ F @ c2
S12 = c1.T @ S @ c2
epsilon1 = c1.T @ F @ c1
epsilon2 = c2.T @ F @ c2
```

Here the fragment coefficients are embedded in the dimer AO basis. F and S are dimer Fock and AO-overlap matrices. The diagonal terms are projected dimer-Fock elements, not substituted isolated-fragment orbital energies. The conversion factor is 27,211.4 meV/Hartree.

Defaults are PW91/6-31G*, density-fitted RKS, dimer quadrature level 2 and tracking quadrature level 1. Optional reference-orbital state selection searches occupied candidates by absolute overlap, adjusts their phase by the overlap sign, and can introduce interpolated intermediate geometries when overlap is low. The overlap threshold is 0.8 and the phase-refinement limit is 10. Inspect the actual implementation for SCF/library defaults, failure branches and warning behavior; a warning does not guarantee a successfully tracked state.

```bash
python quantum/calc_one_frame5_pair_v.py \
  --top /path/to/box.prmtop --traj /path/to/production.nc --frame 5 \
  --res1 396 --res2 486 --phase-ref-resid 1 --backend cpu \
  --out outputs/pair.tsv
```

`calc_frame5_v_batch.py` reuses the single-pair functions for a specified pair-index range. `build_frame5_pkl_subset.py` combines a user-supplied pair list and computed reference results into the archived DM dictionary format. Run each script with `--help` for required ranges and input fields.

The fixed-dimer signed-V learning task assumes a consistent orbital-state/phase reference. The signs of independently evaluated dimers are convention-dependent; the Frame5/Merged learning tasks directly train abs(V). The presence of a phase-reference option in a utility does not establish a gauge-invariant physical meaning for every individual pair sign.
