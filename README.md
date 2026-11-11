# EcPredict

Code accompanying **Euclidean-Threshold Clustering with OPT3D Descriptors for Data-Efficient Prediction of Electronic Couplings in Amorphous Pentacene**.

EcPredict compares distance-matrix (DM) and optimized 3D-MoRSE (OPT3D) representations, random splitting and Euclidean-threshold clustering, and artificial neural network (ANN) predictions of intermolecular electronic couplings.

**This is a code-only release. No research datasets, MD trajectories, prediction records, pretrained models, or manuscript files are included.** Commands below require your own local inputs. Dataset names in the documentation describe the research tasks, not bundled files.

## Scope

| Research task | Prediction target | Sampling and representations |
| --- | --- | --- |
| 1–1297: fixed pentacene dimer, 30,000 configurations | Phase-referenced signed `V` | DM with or without training-only permutation augmentation; OPT3D; random and clustering |
| Frame5: 15,644 distinct dimers from one archived snapshot | Directly trained magnitude `abs(V)` | DM and OPT3D; random and clustering |
| Merged Frames: 61,963 dimer records from four archived snapshots | Directly trained magnitude `abs(V)` | OPT3D; random and clustering |

For `abs(V)` tasks, only reference targets are converted to magnitudes before training. The network has a linear output; predictions are **not** absolutized or clipped afterward.

## Repository structure

```text
EcPredict/
├── README.md
├── requirements.txt            # Lightweight descriptor/plotting dependencies
├── requirements-ml.txt         # CPU-capable ANN dependencies
├── requirements-quantum.txt    # Optional trajectory/quantum dependencies
├── run_experiment.py           # One interface for the archived ANN pipelines
├── descriptors/                # DM construction, OPT3D calculation and merging
├── models/                     # Eight byte-identical research training scripts
├── quantum/                    # Dimer extraction and coupling-reproduction utilities
├── plotting/                   # Parity and learning-curve plots from user-generated outputs
├── configs/ann.json            # Architecture/training settings
├── docs/                       # Methods, data formats, provenance and environment notes
└── tests/                      # Small generated-input checks; no paper data
```

## Installation

Python 3.11 is recommended for the inspected source versions. Use a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Add ANN training when needed:
python -m pip install -r requirements-ml.txt
# Add trajectory extraction and DFT when needed:
python -m pip install -r requirements-quantum.txt
```

GPU PCA uses optional **CuPy, cuDF and cuML**, while GPU quantum calculations use **GPU4PySCF**. Install GPU packages for your CUDA/driver combination in a compatible environment; they are deliberately not installed by the lightweight requirements file. [Environment notes](docs/environment.md) distinguish CPU checks from the original GPU workflows.

## 1. Prepare DM and OPT3D inputs

Input formats are documented in [data formats](docs/data_formats.md). For already unwrapped, spatially assembled dimer XYZ frames and signed DFT labels in meV:

```bash
python descriptors/calculate_dm.py \
  --xyz /path/to/dimers.xyz --targets /path/to/signed_V_meV.npy \
  --output /path/to/dm.pkl

python descriptors/calculate_opt3d.py \
  --input /path/to/dm.pkl --output /path/to/opt3d.pkl \
  --sl 0.1 --nsc 100
```

Each XYZ frame contains monomer 1 (36 atoms) followed by monomer 2 (36 atoms), with a consistent atom order. `calculate_dm.py` does not repair periodic images; use physically assembled dimers. Its default `float16` output follows the archived DM storage, including target rounding; `--dtype float32` preserves more precision but creates a different input representation.

The seven OPT3D blocks are **U, M, V, P, E, IP, RC**. Each contains `nsc` values with `s = 0, ..., nsc-1`. Both intra- and intermolecular unique pairs are included. Defaults are `sl = 0.1`, `nsc = 100`, giving **700 features**. `nsc = 32` gives **224 features**. When slicing an existing array, each block is sliced independently.

To combine snapshots, first calculate them using identical descriptor settings, then concatenate in the intended order:

```bash
python descriptors/merge_opt3d.py \
  --inputs /path/to/frame5.pkl /path/to/frame10000.pkl \
           /path/to/frame20000.pkl /path/to/frame30000.pkl \
  --output /path/to/merged.pkl
```

## 2. Run ANN experiments

Run from the repository root. Each experiment requires a **new/empty output directory** so that cached partitions cannot be reused for a different input.

### Fixed dimer: signed V

```bash
python run_experiment.py \
  --input /path/to/dm_1_1297.pkl --descriptor dm --target signed \
  --sampling random --train-pct 60 --permutations 12 \
  --output outputs/fixed_dm_random60 --require-gpu

python run_experiment.py \
  --input /path/to/dm_1_1297.pkl --descriptor dm --target signed \
  --sampling cluster --train-pct 30 --permutations 12 \
  --output outputs/fixed_dm_cluster30 --require-gpu
```

### Frame5: coupling magnitude

```bash
python run_experiment.py \
  --input /path/to/frame5_opt3d.pkl --descriptor opt3d --target abs \
  --sampling random --train-pct 90 --nsc 100 \
  --output outputs/frame5_opt_random90 --require-gpu

python run_experiment.py \
  --input /path/to/frame5_opt3d.pkl --descriptor opt3d --target abs \
  --sampling cluster --train-pct 20 --nsc 32 \
  --output outputs/frame5_opt32_cluster20 --require-gpu
```

For Merged Frames, pass the merged OPT3D input with `--target abs`. To vary the original training percentage from 10% to 90%, repeat a command with `--train-pct 10`, `20`, ..., `90`. Random splitting uses the integer count `floor(N_clean * pct / 100)`; clustering searches for an approximately matching representative count and records the achieved fraction.

### Matched descriptor thresholds

```bash
python run_experiment.py \
  --input /path/to/frame5_opt3d.pkl --descriptor opt3d --target abs \
  --sampling cluster --threshold 1.0 --nsc 100 \
  --output outputs/frame5_threshold1 --require-gpu
```

The threshold is an **RMS-normalized Euclidean distance in descriptor space**, not a spatial separation between molecules. A new threshold changes both the medoids and the remaining evaluation configurations. See [clustering and evaluation](docs/methods.md).

Use `--dry-run` to inspect the selected source and options without training. `--expect-gpu 'RTX 4090'` additionally checks the available GPU name. `--require-gpu` prevents silent substitution of CPU PCA. Original experiment GPU families are recorded in [source_manifest.json](docs/source_manifest.json).

## 3. Inspect generated outputs

A completed experiment writes:

- `train_predictions.csv` and `test_predictions.csv`: original input indices, signed DFT reference, actual training target, predictions and absolute errors;
- `metrics.json`: actual train/test counts, MAEs, descriptor settings, seeds, input/source hashes, backend and package versions;
- `split_indices.npz`, `preprocessing.npz`, `model.json`, `model_weights.npz`, and `history.csv`.

Training prediction exports contain **original configurations only**, excluding augmented DM copies. The archived pipeline's training MAE, which may include these copies, is reported separately.

## 4. Plot predictions and learning curves

```bash
python plotting/plot_parity.py \
  --run outputs/frame5_opt_random90 --target abs \
  --output outputs/frame5_opt_random90/parity.svg --colorbar

python plotting/plot_learning_curves.py \
  --runs outputs/frame5_opt_random10 outputs/frame5_opt_random20 \
  --x fraction --output outputs/learning_curve.svg
```

Parity plots use closed frames, blue training points, warm-density test crosses, an identity line, and a MAE computed from the exported test records. Scatter coordinates retain negative predictions. Use `--x count` for a learning curve against actual training counts.

## 5. Optional MD/quantum steps

`quantum/` contains the existing argument-driven extraction and coupling-reproduction tools. The minimum intermolecular atom distance criterion includes H atoms and uses a cutoff of 5.0 Å. Quantum defaults are PW91/6-31G*, with a reference-orbital-overlap option for state/phase tracking. See [quantum calculations](docs/quantum.md) before using these utilities, particularly the distinction between archived snapshot labels and zero-based trajectory indices.

```bash
python quantum/find_frame5_dimers.py \
  --top /path/to/box.prmtop --traj /path/to/production.nc \
  --frame 5 --cutoff 5.0 --out outputs/candidates.tsv
```

## Checks and interpretation

```bash
python -m unittest discover -s tests -v
```

The included checks generate temporary inputs; they do not reproduce the paper's full experiments. **The original training procedure uses the same held-out subset for early stopping/learning-rate scheduling and subsequently reports it as Test.** Results therefore characterize the archived evaluation procedure, not a fully independent external test. CPU/GPU PCA, solver selection, numeric precision and GPU architecture can produce different partitions or MAEs. Seeds alone do not ensure bitwise reproduction across backends. The code-only release does not guarantee reproduction of published numerical values without the corresponding input arrays, partitions and environment.

## Related work and citation

OPT3D representation background: Li et al., *A set of optimized 3D-MoRSE descriptors for molecular representation*, npj Computational Materials (2026), and the [OPT3D repository](https://github.com/Leonardo-Lixp/OPT3D). This project implements the seven-weight pentacene workflow used in our EC study; it does not distribute the upstream molecular-property datasets or reproduce its RF/ensemble pipelines.

For intermolecular 3D-MoRSE and orbital-reference methods, see [method references](docs/references.md). A software citation is provided in `CITATION.cff`; final journal/DOI metadata can be added after publication. Reuse licensing should be specified by the authors before the public release.
