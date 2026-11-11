"""Portable entry point around the byte-identical archived ANN pipelines.

Keeps the original splitting/preprocessing/training path and adds original-row
tracking, prediction export, metrics and software/hardware metadata. No research
datasets are distributed with this code-only repository.
"""
from pathlib import Path
from collections import defaultdict, deque
from importlib import metadata
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def row_key(row):
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--descriptor', choices=['dm', 'opt3d'], required=True)
    p.add_argument('--target', choices=['signed', 'abs'], required=True)
    p.add_argument('--sampling', choices=['random', 'cluster'], required=True)
    p.add_argument('--permutations', type=int, default=0)
    p.add_argument('--nsc', type=int, default=100)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--n-train', type=int, help='Exact random-split training count')
    group.add_argument('--train-pct', type=float, help='Original-configuration percentage, 0 < pct < 100')
    group.add_argument('--threshold', type=float, help='Fixed RMS distance in descriptor space')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--config', type=Path, default=ROOT/'configs/ann.json')
    p.add_argument('--output', required=True, type=Path, help='New/empty run directory')
    p.add_argument('--require-gpu', action='store_true', help='Fail rather than using CPU PCA')
    p.add_argument('--expect-gpu', help='Require GPU name to contain this string')
    p.add_argument('--dry-run', action='store_true', help='Validate options and show the selected archived source')
    args = p.parse_args()
    if args.permutations < 0 or (args.descriptor == 'opt3d' and args.permutations):
        p.error('Nonnegative permutations apply to DM only; OPT3D requires zero')
    if args.train_pct is not None and not 0 < args.train_pct < 100:
        p.error('--train-pct must be between 0 and 100')
    if args.nsc < 1 or (args.threshold is not None and args.threshold <= 0):
        p.error('nsc and any threshold must be positive')
    if args.sampling == 'random' and args.threshold is not None:
        p.error('Random splitting requires --n-train or --train-pct')
    if args.sampling == 'cluster' and args.n_train is not None:
        p.error('Clustering requires --threshold or --train-pct')
    return args


def main():
    args = parse_args()
    source = ROOT/'models'/f'{args.descriptor}_{args.sampling}_{args.target}.py'
    input_path, output = args.input.resolve(), args.output.resolve()
    config_path = args.config.resolve()
    if not input_path.is_file() or not config_path.is_file():
        raise FileNotFoundError('Supply an existing input PKL and configuration JSON')
    if args.dry_run:
        print(json.dumps({'pipeline': str(source.relative_to(ROOT)), 'input': str(input_path),
                          'target': args.target, 'sampling': args.sampling,
                          'permutations': args.permutations, 'nsc': args.nsc,
                          'train_pct': args.train_pct, 'n_train': args.n_train,
                          'threshold': args.threshold, 'output': str(output)}, indent=2))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new/empty output directory; split caches must not mix inputs')
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)  # Original logs and split caches are confined to this run.
    os.environ['ML_SEED'] = str(args.seed)
    os.environ['ML_SPLIT_SEED'] = str(42+args.seed)
    spec = importlib.util.spec_from_file_location('archived_pipeline', source)
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    base.GLOBAL_SEED = args.seed
    if hasattr(base, 'SPLIT_SEED'):
        base.SPLIT_SEED = 42+args.seed
    if (args.require_gpu or args.expect_gpu) and base.device_gpu != 1:
        raise RuntimeError('The requested GPU PCA backend is unavailable')
    gpu_name = None
    if base.device_gpu:
        gpu_name = base.cp.cuda.runtime.getDeviceProperties(0)['name']
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
    if args.expect_gpu and args.expect_gpu.replace(' ', '').lower() not in gpu_name.replace(' ', '').lower():
        raise RuntimeError(f'GPU mismatch: got {gpu_name!r}, expected {args.expect_gpu!r}')
    with input_path.open('rb') as f:
        raw = pickle.load(f)
    if args.descriptor == 'dm' and not isinstance(raw, dict):
        raise ValueError('DM input must be a dictionary with distance_matrix and atom_list')
    if args.descriptor == 'opt3d' and isinstance(raw, dict):
        raise ValueError('OPT3D input must be a descriptor array')
    rows = raw['distance_matrix'] if isinstance(raw, dict) else np.asarray(raw)
    source_labels = np.asarray(rows[:, -1], dtype=np.float64).copy()
    del rows, raw
    state = {}

    # Hooks never draw random numbers or change the original split/model path.
    original_clean = base.DataProcessor.clean_data_keep_all_cols
    def clean_and_track(self, rows):
        lookup = defaultdict(deque)
        for i, row in enumerate(rows):
            lookup[row_key(row)].append(i)
        cleaned = original_clean(self, rows)
        state['cleaned_to_input'] = np.asarray([lookup[row_key(r)].popleft() for r in cleaned])
        return cleaned
    base.DataProcessor.clean_data_keep_all_cols = clean_and_track

    splitter = base.RandomSplitter if args.sampling == 'random' else base.ThresholdClusterSplitter
    if args.sampling == 'random':
        original_init = splitter.__init__
        def init(self, n_train, random_state=42):
            return original_init(self, n_train, random_state=42+args.seed)
        splitter.__init__ = init
    original_split = splitter.split
    def capture_split(self, cleaned):
        result = original_split(self, cleaned)
        raw_train, raw_test, tr_idx = result
        remaining = np.setdiff1d(np.arange(len(cleaned)), tr_idx)
        lookup = defaultdict(deque)
        for i in remaining:
            lookup[row_key(cleaned[i])].append(i)
        te_idx = np.asarray([lookup[row_key(row)].popleft() for row in raw_test])
        for name, raw_rows, idx in [('train', raw_train, tr_idx), ('test', raw_test, te_idx)]:
            state[name+'_rows'] = raw_rows.copy()
            state[name+'_indices'] = state['cleaned_to_input'][idx]
        state['group_ids'] = getattr(self, 'group_ids_', None)
        return result
    splitter.split = capture_split

    original_train = base.NeuralNetworkModel.train
    def capture_train(self, x_train, y_train, x_test, y_test):
        state['test_pca'] = x_test
        state['actual_test_target'] = np.asarray(y_test).copy()
        if not args.permutations:
            state['original_train_pca'] = x_train
        return original_train(self, x_train, y_train, x_test, y_test)
    base.NeuralNetworkModel.train = capture_train

    base.set_data_random_seed(args.seed)
    config = base.ModelConfig.load(str(config_path))
    # Keras 3 dropped the legacy string alias; the Huber(delta=1) formula is unchanged.
    import keras
    legacy_loss_alias_mapped = False
    if config.loss == 'huber_loss':
        try:
            keras.losses.get(config.loss)
        except (ValueError, KeyError):
            config.loss = 'huber'
            legacy_loss_alias_mapped = True
    config.save(output/'config.json')
    pipeline = base.MLEPipeline(config)
    if args.sampling == 'random':
        # Count finite/non-NaN rows after original cleaning, without changing its RNG use.
        with input_path.open('rb') as f:
            obj = pickle.load(f)
        arr = np.asarray(obj['distance_matrix'] if isinstance(obj, dict) else obj, dtype=np.float32)
        if args.descriptor == 'opt3d':
            arr = base.DataProcessor.slice_opt3d_nsc(arr, args.nsc)
        count = int(np.sum(~np.isnan(arr).any(axis=1)))
        del arr, obj
        n_train = args.n_train if args.n_train is not None else int(count*args.train_pct/100)
        if not 1 <= n_train < count:
            raise ValueError('Random training count must leave at least one test configuration')
        extra = {'slice_nsc': args.nsc} if args.descriptor == 'opt3d' else {}
        reported_train_mae, reported_test_mae = pipeline.run(input_path, n_train, args.permutations,
                                                           'run', **extra)
    else:
        kwargs = dict(split_method='threshold')
        if args.threshold is None:
            kwargs.update(auto_threshold=True, target_train_pct=args.train_pct,
                          auto_target_pcts=[args.train_pct])
        else:
            kwargs['threshold'] = args.threshold
        if args.descriptor == 'opt3d':
            kwargs['slice_nsc'] = args.nsc
        reported_train_mae, reported_test_mae = pipeline.run(input_path, 0, args.permutations,
                                                           'run', **kwargs)
    reducer = pipeline.pca_reducer
    if args.permutations:
        state['original_train_pca'] = reducer.transform_in_batches(
            state['train_rows'][:, :-1], batch_size=20000, label='original_train_export')
    import pandas as pd
    metrics = {}
    for split in ['train', 'test']:
        idx = state[split+'_indices']
        x = state['original_train_pca'] if split == 'train' else state['test_pca']
        prediction = pipeline.model.predict(x).reshape(-1).astype(np.float64)
        signed = source_labels[idx]
        target = np.abs(signed) if args.target == 'abs' else signed
        raw_target = state[split+'_rows'][:, -1]
        if args.target == 'abs':
            raw_target = np.abs(raw_target)
        if not np.allclose(target, raw_target, rtol=1e-6, atol=1e-6):
            raise ValueError('Row mapping does not match the actual training target')
        if split == 'test' and not np.allclose(target, state['actual_test_target'], rtol=1e-6, atol=1e-6):
            raise ValueError('Exported target differs from the target passed to model training')
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite predictions cannot be exported as completed results')
        error = np.abs(target-prediction)
        metrics[split+'_mae_meV'] = float(error.mean())
        pd.DataFrame({'sample_index': idx, 'V_DFT_meV': signed,
                      'DFT_target_meV': target, 'predicted_meV': prediction,
                      'abs_error_meV': error}).to_csv(output/f'{split}_predictions.csv', index=False)
    assert not np.intersect1d(state['train_indices'], state['test_indices']).size
    if abs(metrics['test_mae_meV']-reported_test_mae) > 1e-3:
        raise ValueError('Exported MAE disagrees with model evaluation')
    cpu = lambda x: base.cp.asnumpy(x) if base.device_gpu else np.asarray(x)
    np.savez_compressed(output/'split_indices.npz', train=state['train_indices'],
                        test=state['test_indices'], cleaned=state['cleaned_to_input'])
    np.savez_compressed(output/'preprocessing.npz', scaler_mean=cpu(reducer.mean_),
                        scaler_std=cpu(reducer.std_), pca_mean=cpu(reducer.pca_.mean_),
                        pca_components=cpu(reducer.pca_.components_))
    (output/'model.json').write_text(pipeline.model.model.to_json())
    np.savez_compressed(output/'model_weights.npz', *pipeline.model.model.get_weights())
    pd.DataFrame(pipeline.model.history.history).to_csv(output/'history.csv', index=False)
    packages = {}
    for name in ['numpy','scikit-learn','tensorflow','keras','cupy-cuda12x','cuml-cu12','pandas']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    metrics.update(descriptor=args.descriptor, target=args.target, sampling=args.sampling,
                   nsc=args.nsc if args.descriptor == 'opt3d' else None,
                   permutations=args.permutations, seed=args.seed, split_seed=42+args.seed,
                   n_train=len(state['train_indices']), n_test=len(state['test_indices']),
                   selected_threshold=getattr(pipeline, 'selected_threshold', None),
                   pipeline_file=str(source.relative_to(ROOT)), source_sha256=sha256(source),
                   input_sha256=sha256(input_path), active_gpu=gpu_name,
                   pca_backend='cuml' if base.device_gpu else 'sklearn', software=packages,
                   reported_augmented_train_mae_meV=reported_train_mae,
                   legacy_huber_alias_mapped=legacy_loss_alias_mapped,
                   evaluation_note='Original workflow: Test subset also used for early stopping and learning-rate scheduling.',
                   prediction_note='Training export contains original configurations only. Predictions are not absolutized or clipped.')
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
