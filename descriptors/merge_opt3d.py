"""Concatenate user-supplied OPT3D arrays in the supplied snapshot order."""
from pathlib import Path
import argparse
import pickle
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    blocks = []
    for path in args.inputs:
        with path.open('rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            raise ValueError('Expected an OPT3D array, not a DM dictionary')
        block = np.asarray(obj)
        if block.ndim != 2 or block.shape[1] < 8 or (block.shape[1]-1) % 7:
            raise ValueError('Expected seven descriptor blocks plus a final target column')
        if blocks and block.shape[1] != blocks[0].shape[1]:
            raise ValueError('Inputs must use identical nsc, sl, weights, and atom-pair scope')
        blocks.append(block)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as f:
        pickle.dump(np.vstack(blocks), f, protocol=4)
    print('Saved', args.output)


if __name__ == '__main__':
    main()
