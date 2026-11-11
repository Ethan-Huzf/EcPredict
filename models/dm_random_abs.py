#!/usr/bin/env python3
"""
Molecular Machine Learning Pipeline - Random Split then Permute then PCA
Corrected workflow:
1. Data cleaning
2. Random split (on original features)
3. Permute training data (on original features)
4. PCA fit on training data, transform both train and test
5. Train and test model

Usage:
    python kimi_random_pca.py <input.pkl> <n_train_samples> [total_perms] [output_name]
"""

import sys
import time
import os
import logging
import argparse
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import itertools
import random
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 统一使用同一个随机种子，避免不同随机源各自写死导致复现实验时含义不清。
GLOBAL_SEED = 0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('kimi_random_pca.log')
    ]
)
logger = logging.getLogger(__name__)

# GPU/CPU Configuration
device_gpu = 1

if device_gpu == 1:
    try:
        import cudf
        from cuml.decomposition import PCA
        import cupy as cp
        logger.info("GPU libraries (cuML, cuPy) loaded successfully")
    except ImportError as e:
        logger.warning(f"GPU libraries not available: {e}. Falling back to CPU.")
        device_gpu = 0

if device_gpu == 0:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    logger.info("Using CPU for processing")


def set_data_random_seed(seed: int = GLOBAL_SEED):
    """Set random seeds used before TensorFlow model construction."""
    # 固定 Python/NumPy/CuPy 的随机源，让清洗 shuffle、置换 shuffle 和增强后 shuffle 尽量可重复。
    np.random.seed(seed)
    random.seed(seed)
    if device_gpu == 1:
        cp.random.seed(seed)
    logger.info(f"Data random seeds fixed to {seed}")


@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters."""
    lr: float = 0.001
    rate: float = 0.0
    epochs: int = 20000
    batch_size: int = 2048
    activation: str = 'relu'
    loss: str = 'huber_loss'
    shuffle: bool = True
    patience: int = 200
    min_delta: float = 0.0001
    l2_reg: float = 1e-4
    
    layer_ratios: List[int] = None
    max_units: List[int] = None
    
    def __post_init__(self):
        if self.layer_ratios is None:
            self.layer_ratios = [2, 3, 2, 1]
        if self.max_units is None:
            self.max_units = [1024, 2048, 1024, 512]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class DataProcessor:
    """Handles data loading, augmentation, and preprocessing."""
    
    def __init__(self, total_perms: int = 100):
        self.total_perms = total_perms
        logger.info(f"DataProcessor initialized with total_perms={total_perms}")
    
    def get_swaps(self, symbols: List[str]) -> Tuple[Dict, Dict]:
        """Get swap indices for identical atom symbols."""
        from itertools import islice, permutations
        
        symbol_to_indices = {}
        for i, s in enumerate(symbols):
            symbol_to_indices.setdefault(s, []).append(i)
        
        swappable_groups = {
            sym: indices
            for sym, indices in symbol_to_indices.items()
            if len(indices) > 1
        }
        n_swap_groups = len(swappable_groups)

        if n_swap_groups == 0 or self.total_perms <= 0:
            return {}, symbol_to_indices

        # 将 total_perms 尽量平均分配到各个可置换原子组，余数依次补给前面的原子组。
        base_per_group, remainder = divmod(self.total_perms, n_swap_groups)
        logger.info(
            f"total_perms={self.total_perms}, swappable_groups={n_swap_groups}, "
            f"base_per_group={base_per_group}, remainder={remainder}"
        )

        swap_indices = {}
        for group_id, (sym, indices) in enumerate(swappable_groups.items()):
            group_perms = base_per_group + (1 if group_id < remainder else 0)
            if group_perms <= 0:
                continue
            # 每个组分到的是非恒等置换数；permutations 的第一个结果是恒等排列，所以多取 1 个候选。
            perms = list(islice(permutations(indices), group_perms + 1))
            random.shuffle(perms)
            swap_indices[sym] = perms
        
        return swap_indices, symbol_to_indices
    
    def load_original_data(self, input_path: Path):
        """Load original dictionary format data."""
        logger.info(f"Loading original data from {input_path}")
        try:
            with open(input_path, 'rb') as f:
                alldata = pickle.load(f)
            tdata = alldata["distance_matrix"].astype(np.float32)
            atom_symbols = alldata["atom_list"]
            logger.info(f"Loaded data shape: {tdata.shape}")
            return tdata, atom_symbols
        except Exception as e:
            logger.error(f"Failed to load original data: {e}")
            raise
    
    def generate_permutation_data(self, tdata, atom_symbols) -> np.ndarray:
        """Generate permuted data based on atom symbol swaps."""
        distance_matrix_v = tdata
        indx, oldindx = self.get_swaps(atom_symbols)
        natoms = len(atom_symbols)
        
        all_indx = []
        for element in indx:
            for new_index in indx[element]:
                # 跳过没有改变原子顺序的恒等排列，只保留真正发生交换的置换。
                if not np.array_equal(oldindx[element], np.array(new_index)):
                    original_cols = oldindx[element]
                    new_order = new_index
                    all_indices = list(range(natoms))
                    new_indices = []
                    for i in all_indices:
                        if i in original_cols:
                            new_indices.append(original_cols[new_order.index(i)])
                        else:
                            new_indices.append(i)
                    all_indx.append(new_indices)
        
        permuted_dms = []
        for i in distance_matrix_v:
            v = i[-1]
            dis = i[:-1].reshape(natoms, natoms)
            for new_indices in all_indx:
                # 距离矩阵的行和列都代表原子编号，置换原子顺序时必须两轴同步重排，才能保持矩阵物理含义正确。
                new_dis = dis[np.ix_(new_indices, new_indices)]
                flatted = new_dis.flatten()
                permuted_dms.append(np.hstack((flatted, v)))
        
        data = np.array(permuted_dms, dtype=np.float32)
        tdata = np.vstack((distance_matrix_v, data))
        np.random.shuffle(tdata)
        
        logger.info(f"Permutation data generated, shape: {tdata.shape}")
        return tdata
    
    def clean_data_keep_all_cols(self, tdata: np.ndarray) -> np.ndarray:
        """Remove NaN values only, keep all columns for permutation."""
        logger.info("Cleaning data: removing NaN only (keeping all columns for permutation)")
        
        if device_gpu == 1:
            tdata_gpu = cp.array(tdata, dtype=cp.float32)
            cp.random.shuffle(tdata_gpu)
            mask = ~cp.any(cp.isnan(tdata_gpu), axis=1)
            data_cleaned = tdata_gpu[mask]
            result = data_cleaned.get()
            del tdata_gpu, data_cleaned
        else:
            tdata = tdata.astype(np.float32)
            np.random.shuffle(tdata)
            mask = ~np.any(np.isnan(tdata), axis=1)
            result = tdata[mask]
        
        logger.info(f"Cleaned data shape: {result.shape}")
        return result
    
    def clean_data(self, tdata: np.ndarray, valid_cols: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Remove NaN values and duplicate columns."""
        logger.info("Cleaning data: removing NaN and duplicates")
        
        if device_gpu == 1:
            tdata_gpu = cp.array(tdata, dtype=cp.float32)
            cp.random.shuffle(tdata_gpu)
            mask = ~cp.any(cp.isnan(tdata_gpu), axis=1)
            data_no_nan = tdata_gpu[mask]
            
            if valid_cols is None:
                data_transposed = data_no_nan.T
                unique_cols, indices = cp.unique(data_transposed, axis=0, return_index=True)
                valid_cols = cp.sort(indices).get()
            
            data_cleaned = data_no_nan[:, valid_cols.tolist()]
            cp.random.shuffle(data_cleaned)
            
            result = data_cleaned.get()
            del tdata_gpu, data_no_nan, data_cleaned
        else:
            tdata = tdata.astype(np.float32)
            np.random.shuffle(tdata)
            mask = ~np.any(np.isnan(tdata), axis=1)
            data_no_nan = tdata[mask]
            
            if valid_cols is None:
                data_transposed = data_no_nan.T
                unique_cols, indices = np.unique(data_transposed, axis=0, return_index=True)
                valid_cols = np.sort(indices)
            
            data_cleaned = data_no_nan[:, valid_cols]
            np.random.shuffle(data_cleaned)
            
            result = data_cleaned
        
        logger.info(f"Cleaned data shape: {result.shape}, valid columns: {len(valid_cols)}")
        return result, valid_cols


class RandomSplitter:
    """Handles random train/test split."""
    
    def __init__(self, n_train: int, random_state: int = 42):
        self.n_train = n_train
        self.random_state = random_state
    
    def split(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data randomly into train and test sets.
        
        Returns:
            (train_data, test_data, train_indices)
        """
        n_samples = len(data)
        
        # Validate n_train
        if self.n_train >= n_samples:
            logger.warning(f"n_train ({self.n_train}) >= n_samples ({n_samples}), using all data for training")
            self.n_train = n_samples - 1
        if self.n_train < 1:
            logger.warning(f"n_train ({self.n_train}) < 1, using 1 sample for training")
            self.n_train = 1
        
        indices = np.arange(n_samples)
        
        # Use sklearn's train_test_split with train_size as integer
        train_indices, test_indices = train_test_split(
            indices,
            train_size=self.n_train,
            random_state=self.random_state,
            shuffle=True
        )
        
        train_data = data[train_indices]
        test_data = data[test_indices]
        
        logger.info(f"Random split: Train={len(train_data)}, Test={len(test_data)}")
        logger.info(f"Train ratio: {len(train_data)/n_samples:.3f}, Test ratio: {len(test_data)/n_samples:.3f}")
        
        return train_data, test_data, train_indices


class PCAReducer:
    """Handles PCA dimensionality reduction."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_components_ = None
        self.pca_ = None
    
    @staticmethod
    def gpu_standard_scaler(data: 'cp.ndarray') -> 'cp.ndarray':
        """Standardize data on GPU."""
        mean = cp.mean(data, axis=0)
        std = cp.std(data, axis=0)
        std[std == 0] = 1
        # 完整矩阵在大训练比例和高置换数时很占显存；原地标准化避免额外生成一份同样大小的 X_scaled。
        data -= mean
        data /= std
        return data, mean, std
    
    @staticmethod
    def get_mle_n_components(X_gpu: 'cp.ndarray', max_components: int = 1000) -> int:
        """Calculate optimal PCA components using Minka's MLE with limit."""
        n_samples, n_features = X_gpu.shape
        n_components = min(n_samples, n_features, max_components)
        
        pca = PCA(n_components=n_components, svd_solver='full')
        pca.fit(X_gpu)
        
        d = pca.explained_variance_
        mle_scores = cp.zeros(len(d))
        
        for k in range(1, len(d)):
            v = cp.mean(d[k:])
            if v <= 0:
                break
            part1 = -0.5 * cp.sum(cp.log(d[:k]))
            part2 = -0.5 * (n_features - k) * cp.log(v)
            mle_scores[k] = part1 + part2
        
        best_k = cp.argmax(mle_scores[1:]).item() + 1
        logger.info(f"MLE estimated optimal components: {best_k}")
        return best_k
    
    def fit(self, X_init: np.ndarray):
        """Fit PCA on training data only."""
        logger.info("Fitting PCA on training data...")
        
        n_samples, n_features = X_init.shape
        
        if device_gpu == 1:
            X_gpu = cp.array(X_init, dtype=cp.float32)
            X_scaled, self.mean_, self.std_ = self.gpu_standard_scaler(X_gpu)
            
            self.n_components_ = self.get_mle_n_components(X_scaled)
            self.pca_ = PCA(n_components=self.n_components_, svd_solver='full')
            self.pca_.fit(X_scaled)
            
            del X_gpu, X_scaled
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_init)
            self.mean_ = scaler.mean_
            self.std_ = scaler.scale_
            
            if n_samples >= n_features:
                pca = PCA(n_components='mle')
            else:
                n_components = min(n_samples, n_features) - 1
                logger.info(f"n_samples ({n_samples}) < n_features ({n_features}), using {n_components} components")
                pca = PCA(n_components=n_components)
            
            pca.fit(X_scaled)
            self.pca_ = pca
            self.n_components_ = pca.n_components_
        
        logger.info(f"PCA fitted with {self.n_components_} components")
    
    def transform(self, X_init: np.ndarray) -> np.ndarray:
        """Transform data using fitted PCA."""
        if self.pca_ is None:
            raise ValueError("PCA must be fitted before transform")
        
        logger.info("Transforming data with fitted PCA...")
        
        if device_gpu == 1:
            X_gpu = cp.array(X_init, dtype=cp.float32)
            std = self.std_.copy()
            std[std == 0] = 1
            X_scaled = (X_gpu - self.mean_) / std
            X_reduced = self.pca_.transform(X_scaled)
            X_result = X_reduced.get()
            del X_gpu, X_scaled, X_reduced
        else:
            from sklearn.preprocessing import StandardScaler
            X_scaled = (X_init - self.mean_) / self.std_
            X_result = self.pca_.transform(X_scaled)
        
        logger.info(f"PCA transform complete. Shape: {X_result.shape}")
        return X_result

    def transform_in_batches(self, X_init: np.ndarray, batch_size: int = 20000, label: str = "data") -> np.ndarray:
        """Transform data by batches using the already fitted PCA."""
        if self.pca_ is None:
            raise ValueError("PCA must be fitted before transform")

        n_samples = X_init.shape[0]
        if batch_size <= 0 or n_samples <= batch_size:
            return self.transform(X_init)

        logger.info(f"Transforming {label} data in batches: total={n_samples}, batch_size={batch_size}")
        X_result = None

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            logger.info(f"Transforming {label} batch: rows {start}:{end}")

            # PCA 已经完整 fit；这里分批只降低 transform 阶段的峰值显存，不改变 PCA 结果。
            X_batch = self.transform(X_init[start:end])
            if X_result is None:
                X_result = np.empty((n_samples, X_batch.shape[1]), dtype=X_batch.dtype)
            X_result[start:end] = X_batch
            del X_batch

            if device_gpu == 1:
                # 释放当前批次产生的 CuPy 临时显存块，避免后续批次显存逐步累积。
                cp.get_default_memory_pool().free_all_blocks()

        logger.info(f"Batched PCA transform complete for {label}. Shape: {X_result.shape}")
        return X_result


class NeuralNetworkModel:
    """Neural network model for regression."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.history = None
        self._setup_gpu()
    
    def _setup_gpu(self):
        """Configure GPU settings."""
        import tensorflow as tf
        
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
        os.environ['TF_FORCE_GPU_ALLOCATION'] = '1'
        # 固定 TensorFlow/Keras 的随机源，主要影响模型初始化和训练过程中的随机性。
        tf.keras.utils.set_random_seed(GLOBAL_SEED)
        try:
            # 能启用时使用确定性算子，减少同一参数重复运行的波动。
            tf.config.experimental.enable_op_determinism()
            logger.info("TensorFlow deterministic ops enabled")
        except Exception as e:
            logger.warning(f"TensorFlow deterministic ops not enabled: {e}")
        
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        logger.info(f"Num GPUs Available: {len(gpus)}")
        
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.error(f"GPU configuration error: {e}")
    
    def build_model(self, input_dim: int) -> 'keras.Model':
        """Build the neural network architecture."""
        import keras
        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization, Activation
        from keras import regularizers
        
        units = [
            min(r * input_dim, max_u) 
            for r, max_u in zip(self.config.layer_ratios, self.config.max_units)
        ]
        
        model = Sequential([
            Dense(units[0], input_dim=input_dim, 
                  kernel_regularizer=regularizers.l2(self.config.l2_reg)),
            BatchNormalization(),
            Activation(self.config.activation),
            Dense(units[1]),
            BatchNormalization(),
            Activation(self.config.activation),
            Dense(units[2]),
            BatchNormalization(),
            Activation(self.config.activation),
            Dense(units[3]),
            BatchNormalization(),
            Activation(self.config.activation),
            Dense(1, dtype='float32')
        ])
        
        model.compile(
            loss=self.config.loss,
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config.lr,
                beta_1=0.9, beta_2=0.999, epsilon=1e-10
            ),
            metrics=['mae'],
            jit_compile=True
        )
        
        self.model = model
        logger.info(f"Model built with input_dim={input_dim}")
        model.summary(print_fn=logger.info)
        return model
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              x_test: np.ndarray, y_test: np.ndarray) -> 'keras.callbacks.History':
        """Train the model."""
        import keras
        import tensorflow as tf
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=self.config.min_delta,
            patience=self.config.patience,
            verbose=0,
            restore_best_weights=True
        )
        
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        logger_callback = EpochLogger(interval=10).get_callback()
        
        self.history = self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=0,
            callbacks=[early_stopping, logger_callback, lr_scheduler],
            # 显式使用配置里的 shuffle，避免依赖 Keras 默认值造成复现实验时含义不清。
            shuffle=self.config.shuffle
        )
        
        return self.history
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate model performance."""
        loss, mae = self.model.evaluate(x, y, verbose=0)
        return loss, mae
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(x, verbose=0)


class EpochLogger:
    """Custom callback for logging training progress."""
    
    def __init__(self, interval: int = 100):
        self.interval = interval
    
    def get_callback(self):
        """Return the actual Keras callback."""
        import keras
        
        class _EpochLogger(keras.callbacks.Callback):
            def __init__(inner_self, interval: int = 100):
                super().__init__()
                inner_self.interval = interval
            
            def on_epoch_end(inner_self, epoch: int, logs: Optional[Dict] = None):
                if (epoch + 1) % inner_self.interval == 0:
                    val_loss = logs.get('val_loss', 'N/A') if logs else 'N/A'
                    loss = logs.get('loss', 'N/A') if logs else 'N/A'
                    logger.info(f"Epoch {epoch + 1}: loss = {loss:.4f}, val_loss = {val_loss}")
        
        return _EpochLogger(self.interval)


class MLEPipeline:
    """Main pipeline orchestrating the entire ML workflow."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.data_processor = None
        self.random_splitter = None
        self.pca_reducer = PCAReducer()
        self.model = None
    
    def run(self, input_path: Path, n_train: int = 1000,
            total_perms: int = 100, output_name: Optional[str] = None):
        """Execute the full pipeline with correct workflow:
        1. Data cleaning (remove NaN only, keep all columns for permutation)
        2. Random split
        3. Permute training data (on original features)
        4. PCA fit on training data, transform both
        5. Train and test model
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Starting MLE Pipeline (Clean -> Random Split -> Permute -> PCA)")
        logger.info("=" * 60)

        self.data_processor = DataProcessor(total_perms=total_perms)
        self.random_splitter = RandomSplitter(n_train=n_train)

        # Step 1: Load original data
        data, atom_symbols = self.data_processor.load_original_data(input_path)
        filename = output_name or f"{input_path.stem}_random{n_train}"

        natoms = int(np.sqrt(data.shape[1] - 1))
        logger.info(f"Detected natoms={natoms} from data shape")

        # Step 2: Clean data - remove NaN only, keep all columns for permutation
        logger.info("\nStep 1: Cleaning data (remove NaN only)...")
        data_clean = self.data_processor.clean_data_keep_all_cols(data)
        logger.info(f"Cleaned data shape: {data_clean.shape}")

        # Step 3: Random split on cleaned data
        logger.info("\nStep 2: Performing random split...")
        train_data_raw, test_data_raw, train_indices = self.random_splitter.split(data_clean)

        logger.info(f"Train raw shape: {train_data_raw.shape}, Test raw shape: {test_data_raw.shape}")

        # Step 4: Permute training data only (on original features)
        if total_perms > 0:
            logger.info("\nStep 3: Generating permutations for training data...")
            train_data = self.data_processor.generate_permutation_data(train_data_raw, atom_symbols)
        else:
            train_data = train_data_raw

        # Step 5: Split into X and y
        logger.info("Using abs(V) target: y = |last column|")
        X_train_orig = train_data[:, :-1]
        y_train = np.abs(train_data[:, -1])
        X_test_orig = test_data_raw[:, :-1]
        y_test = np.abs(test_data_raw[:, -1])

        # Step 6: PCA - fit on training data only, then transform both
        logger.info("\nStep 4: Applying PCA (fit on train, transform both)...")
        self.pca_reducer = PCAReducer()
        self.pca_reducer.fit(X_train_orig)
        # 保持 PCA fit 流程不变，只把 train/test 的 transform 拆成小批次以降低峰值显存。
        X_train = self.pca_reducer.transform_in_batches(X_train_orig, batch_size=20000, label="train")
        X_test = self.pca_reducer.transform_in_batches(X_test_orig, batch_size=20000, label="test")

        logger.info(f"Final train shape: {X_train.shape}, test shape: {X_test.shape}")

        # Step 7: Build and train model
        self.model = NeuralNetworkModel(self.config)
        self.model.build_model(X_train.shape[1])
        self.model.train(X_train, y_train, X_test, y_test)

        # Step 8: Evaluate
        _, train_mae = self.model.evaluate(X_train, y_train)
        _, test_mae = self.model.evaluate(X_test, y_test)
        logger.info(f'Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}')

        elapsed = (time.time() - start_time) / 60
        logger.info("=" * 60)
        logger.info(f"Pipeline complete. Total time: {elapsed:.2f} minutes")
        logger.info("=" * 60)

        return train_mae, test_mae


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Molecular Machine Learning Pipeline - Random Split then Permute then PCA'
    )
    parser.add_argument('input_file', type=str, help='Input pickle file')
    parser.add_argument('n_train', type=int, 
                        help='Number of samples in training set')
    parser.add_argument('total_perms', type=int, nargs='?', default=100,
                        help='Total non-identity permutations per sample (0 for no permutation)')
    parser.add_argument('output_name', type=str, nargs='?', default=None,
                        help='Output filename prefix')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration JSON file')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        config = ModelConfig.load(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = ModelConfig()
        logger.info("Using default configuration")
    
    # Save configuration for reproducibility
    config.save('kimi_random_pca_config.json')
    
    # Run pipeline
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    pipeline = MLEPipeline(config)
    train_mae, test_mae = pipeline.run(input_path, args.n_train, args.total_perms, args.output_name)
    
    # Print result in a format easy to parse
    print(f"\n{'='*60}")
    print(f"RESULT_SUMMARY: n_train={args.n_train}, total_perms={args.total_perms}")
    print(f"Train_MAE: {train_mae:.4f}")
    print(f"Test_MAE: {test_mae:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    set_data_random_seed(GLOBAL_SEED)
    main()
