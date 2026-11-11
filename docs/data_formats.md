# Input and output formats

No data files are included. All input paths refer to files supplied by the user. Load pickle files only from trusted sources.

## DM dictionary

```python
{
    'distance_matrix': array_of_shape_N_by_5185,
    'atom_list': ['m1C', 'm1H', ..., 'm2C', 'm2H', ...]
}
```

- The first 5,184 columns are a row-major flattened 72 × 72 distance matrix in Å.
- The final column is the signed reference EC in **meV**, not Hartree.
- `atom_list` has 72 entries and preserves the original atom ordering. Permutation groups are `m1C`, `m1H`, `m2C`, `m2H`; no permutations exchange atoms between monomers or elements.
- Archived DM arrays use `float16`, including quantization of the target column. The training code converts to `float32` but cannot restore precision already lost in storage.
- A DM consists of two 36-atom pentacene monomers (22 C + 14 H each), already assembled in the intended periodic image.

## OPT3D array

An ordinary 2D NumPy array serialized with pickle, shape `(N, 7*nsc + 1)`:

```text
U[0:nsc], M[0:nsc], V[0:nsc], P[0:nsc], E[0:nsc], IP[0:nsc], RC[0:nsc], signed_V_meV
```

Default `nsc=100` gives 701 columns; `nsc=32` gives 225 columns. The final target stays signed in the stored input. The `abs` training scripts apply `abs` to reference labels before fitting. A larger-nsc input is sliced block by block, not by taking the first `7*nsc` columns.

The arrays do not encode `sl` in their numerical shape; ensure `sl=0.1`, matching weighting constants, atom-pair scope and units yourself. Concatenation cannot infer whether two equally shaped arrays were made with different kernels.

## Prediction exports

| Column | Meaning |
| --- | --- |
| `sample_index` | Zero-based row in the user-supplied input before cleaning/shuffling |
| `V_DFT_meV` | Stored signed reference label |
| `DFT_target_meV` | Signed V or abs(V), according to the trained task |
| `predicted_meV` | Actual linear-network output; no post-hoc magnitude or clipping |
| `abs_error_meV` | abs(DFT_target_meV - predicted_meV) |

The training export includes one point per original training configuration. Split indices exclude rows removed by the original NaN cleaning. Identical rows are matched using queues in the row-tracking wrapper.

For a magnitude task, the parity coordinates are `(abs(V_DFT), prediction_of_absV)`. Their MAE is the mean absolute difference of these two columns, not the error against the signed label. Negative network predictions are retained in the exported results.
