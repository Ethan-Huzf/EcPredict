# Environment and numerical reproducibility

## Local code checks

The release-preparation environment has Python 3.11, NumPy 1.26.0, scikit-learn 1.4.1.post1, TensorFlow 2.16.1, Keras 3.2.1, pandas 2.3.3, Matplotlib 3.10.0, RDKit 2023.9.5, PySCF 2.8.0 and MDAnalysis 2.9.0. These versions describe the available checking environment, not a claim that all paper training runs used this exact environment.

The core requirements pin the inspected versions where practical. GPU dependencies are optional and platform-specific. Do not install both incompatible CPU/GPU package variants blindly into an existing research environment. TensorFlow GPU visibility and cuML PCA availability are separate checks.

## Archived training sources

The eight scripts under `models/` were copied byte-for-byte from the per-run `source_pipeline.py` snapshots used by the existing parity-plot export project. Each was checked against that run's recorded SHA-256. `source_manifest.json` records their origin and associated GPU family. Those GPU families are provenance for reference runs, not software requirements or a guarantee of reproducing a specific MAE.

## Wrapper adaptations

`run_experiment.py` does not rewrite the archived scripts. It adds:

- command-line target/descriptor/sampling selection and an explicit original training count or percentage;
- isolated output directories to avoid stale cache reuse between inputs;
- row-index hooks and exports, without consuming additional random numbers in the original split path;
- optional GPU-name/backend checks;
- recorded source/input hashes, seeds, package versions and backend;
- compatibility mapping of the legacy loss string `huber_loss` to `huber` when Keras no longer accepts that alias; both select Huber(delta=1).

For seed 0, the research source's default seed convention is retained. Random-split runs use splitting seed `42 + seed`. Nonzero-seed wrapper runs are new seeded experiments; do not assume they duplicate a particular archived repeated-run task.

CPU fallback is useful for checking installation and running small local examples. It is not a faithful substitute for an archived GPU PCA/shuffle environment. For GPU reproduction, pass `--require-gpu` and, when appropriate, `--expect-gpu`. Record GPU model, CUDA/cuML/TensorFlow/Keras versions and numeric precision alongside every run. Current installed-library versions, kernels and precision may still differ from the archive.

The preparation checks do not rerun all 10–90% experiments, long DFT calculations or the full MD simulation. No claim of bitwise or full numerical reproduction is made from syntax checks alone.
