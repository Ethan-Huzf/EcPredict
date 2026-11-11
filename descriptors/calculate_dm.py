"""Build the research DM-dictionary format from user-supplied dimer XYZ frames."""
from pathlib import Path
import argparse
import pickle
import numpy as np


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    frames, symbols = [], None
    pos = 0
    while pos < len(lines):
        if not lines[pos].strip():
            pos += 1
            continue
        count = int(lines[pos]); block = lines[pos+2:pos+2+count]
        if len(block) != count:
            raise ValueError('Incomplete XYZ frame')
        tokens = [line.split() for line in block]
        current = [row[0] for row in tokens]
        if symbols is not None and current != symbols:
            raise ValueError('Every XYZ frame must use the same atom ordering')
        symbols = current
        frames.append(np.asarray([[float(v) for v in row[1:4]] for row in tokens]))
        pos += count+2
    if not frames:
        raise ValueError('No XYZ frames found')
    return symbols, np.stack(frames)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xyz', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True, help='NPY array of signed DFT V, in meV')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dtype', choices=['float16', 'float32', 'float64'], default='float16',
                   help='Archive DM PKLs use float16, including the stored target column')
    args = p.parse_args()
    symbols, frames = read_xyz(args.xyz)
    if len(symbols) != 72:
        p.error('Expected 72 atoms: monomer 1 first, followed by monomer 2')
    for half in [symbols[:36], symbols[36:]]:
        if half.count('C') != 22 or half.count('H') != 14:
            p.error('Each 36-atom block must be one C22H14 pentacene monomer')
    targets = np.load(args.targets, allow_pickle=False).reshape(-1)
    if len(targets) != len(frames) or not np.isfinite(targets).all() or not np.isfinite(frames).all():
        p.error('Coordinates and one finite target per frame are required')
    rows = np.empty((len(frames), 72*72+1), dtype=args.dtype)
    for i, xyz in enumerate(frames):
        dm = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        rows[i, :-1] = dm.reshape(-1)
    rows[:, -1] = targets
    if not np.isfinite(rows).all():
        p.error('Chosen dtype cannot represent these values')
    atoms = [('m1' if i < 36 else 'm2')+e for i, e in enumerate(symbols)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as f:
        pickle.dump({'distance_matrix': rows, 'atom_list': atoms}, f, protocol=4)
    print(f'Saved {rows.shape}: {args.output}')


if __name__ == '__main__':
    main()
