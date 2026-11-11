"""Convert an existing DM-dictionary pickle to seven-block pentacene OPT3D."""
from pathlib import Path
import argparse
import pickle
import numpy as np
try:
    from .atomic_weights import ATOMIC_WEIGHTS, WEIGHT_NAMES
except ImportError:
    from atomic_weights import ATOMIC_WEIGHTS, WEIGHT_NAMES


def opt3d_from_dm(dm, atom_groups, nsc=100, sl=0.1):
    """Sum w_i*w_j*sin(s*sl*r_ij)/(s*sl*r_ij), i<j, in U/M/V/P/E/IP/RC order.

    The s=0 term is one before weighting. Both intra- and intermolecular
    atom pairs are included. Constants and block order match the archived
    distance-PKL converter. This vectorized implementation avoids Python's
    innermost scattering-parameter loop.
    """
    dm = np.asarray(dm, dtype=np.float64)
    if nsc < 1 or sl <= 0 or not np.isfinite(sl):
        raise ValueError('nsc must be positive and sl must be finite and positive')
    n = len(atom_groups)
    if dm.shape != (n, n) or not np.isfinite(dm).all():
        raise ValueError('Expected a finite square DM matching atom_groups')
    if not np.allclose(dm, dm.T, atol=1e-6):
        raise ValueError('DM must be symmetric')
    elements = [str(a)[2:] if str(a).startswith(('m1', 'm2')) else str(a)
                for a in atom_groups]
    try:
        weights = np.asarray([ATOMIC_WEIGHTS[e] for e in elements])
    except KeyError as exc:
        raise ValueError('This pentacene release supports only C and H') from exc
    i, j = np.triu_indices(n, 1)
    phase = dm[i, j, None] * sl * np.arange(nsc)[None, :]
    kernel = np.sinc(phase / np.pi)
    return ((weights[i] * weights[j]).T @ kernel).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Trusted DM-dictionary PKL')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nsc', type=int, default=100)
    parser.add_argument('--sl', type=float, default=0.1)
    args = parser.parse_args()
    with args.input.open('rb') as f:
        data = pickle.load(f)
    atoms = data['atom_list']
    rows = np.asarray(data['distance_matrix'])
    if rows.ndim != 2 or rows.shape[1] != len(atoms)**2 + 1:
        raise ValueError('DM input must contain flattened distances followed by V in meV')
    output = np.empty((len(rows), len(WEIGHT_NAMES)*args.nsc + 1), dtype=np.float64)
    for k, row in enumerate(rows):
        output[k, :-1] = opt3d_from_dm(row[:-1].reshape(len(atoms), len(atoms)), atoms,
                                      args.nsc, args.sl)
        output[k, -1] = row[-1]  # Retain signed reference; target choice is made at training.
        if (k+1) % 100 == 0:
            print(f'{k+1}/{len(rows)}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as f:
        pickle.dump(output, f, protocol=4)
    print(f'Saved {output.shape}: {args.output}')


if __name__ == '__main__':
    main()
