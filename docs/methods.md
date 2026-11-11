# Clustering, preprocessing and evaluation

## Descriptor-space distance

The research threshold is

```text
RMS(x, y) = sqrt(sum((x[k] - y[k])**2) / d)
```

where `d` is the number of descriptor features. The corresponding unnormalized Euclidean distance is `RMS * sqrt(d)`. For an all-atom 72 × 72 DM, `d=5184`, so this factor is 72. For seven-block OPT3D with `nsc=100`, `d=700`. These are descriptor distances; OPT3D thresholds are not intermolecular separations in Å. Distance thresholds are applied before standardization/PCA.

## Actual clustering rule

1. Apply the original seeded shuffle and remove rows containing NaNs. EC labels are excluded from all distance calculations.
2. Make the first configuration a fixed group leader.
3. Process remaining configurations in that order. Assign a configuration to the closest leader if its RMS distance is at most the threshold; otherwise create a new group and leader. Leaders are not moved.
4. For each completed group, select an actual member minimizing the sum of RMS distances to all group members (the medoid). The numerical implementation uses blocked pairwise distances.
5. Use one medoid per group for training and all other members for evaluation. A singleton contributes one training configuration and no evaluation configurations.

This is **leader-threshold grouping followed by medoid selection**, not iterative global k-medoids optimization. The threshold constrains distance to the original leader during grouping, not every within-group pair or the final medoid. Input order matters.

The archived automatic threshold search builds a full pairwise RMS-distance matrix and uses the original bracket/bisection search for a target leader count. The achieved count may differ slightly from the requested training percentage. Because leader counts are order-dependent, bisection is a practical numerical search rather than a general proof of a unique monotonic threshold–count relation. Large searches require O(N²) distance storage; use a fixed threshold when already known. For N=61,963, just the float32 full distance matrix is approximately 15.36 GB, before workspace overhead.

## Permutation augmentation

After splitting, only DM training configurations are augmented. The archived routine distributes the requested P additional nonidentity permutations across atom-index groups and reindexes both DM rows and columns consistently. It does not generate new molecular geometries or new quantum labels. All augmented versions keep their original label. This finite augmentation is not a guarantee of invariance to every possible atom permutation. OPT3D is not augmented by this DM routine.

## ANN and preprocessing

The scaler and PCA are fitted on the training representations only. Four dense layers use widths

```text
min(2*d',1024), min(3*d',2048), min(2*d',1024), min(d',512)
```

where `d'` is the selected PCA dimension. Each hidden layer is followed by batch normalization and ReLU. The output is one linear neuron. Defaults include Huber loss, Adam learning rate 0.001, batch size 2048 and up to 20,000 epochs. Full settings are in `configs/ann.json` and the source modules.

The wrapper preserves the archived CPU/GPU PCA branches. They are not identical algorithms: the CPU branch uses scikit-learn's selection procedure, while the GPU branch uses the research cuML eigenvalue-based component heuristic, with source-dependent full/Jacobi solver choices. Some DM sources use GPU shuffling, whereas OPT3D sources use the seeded NumPy shuffle. A switch of backend can change row order as well as preprocessing and fitted model.

## Evaluation limitations

The archived `Test` subset is also passed as validation data to early stopping and learning-rate scheduling. It is not a fully independent final test set. The public wrapper retains this procedure to make the research workflow explicit rather than silently replacing it with a different experiment. An independent hold-out design is needed for a separate generalization claim.

Changing a threshold changes both representatives and the remaining evaluation subset. MAEs at different thresholds, across descriptors, or under random and clustering splits are therefore conditional on their actual subsets. A rising DM clustering MAE should not be interpreted as evidence that more training configurations intrinsically worsen an otherwise identical task. Likewise, equal training fractions do not imply equal labeling counts or configuration coverage across datasets.

For new externally assessed studies, reserve an independent test set before representative selection, tune stopping on a separate validation set, and compare sampling strategies on that common final test set. Such a redesign will produce results different from this archive and should be reported separately.
