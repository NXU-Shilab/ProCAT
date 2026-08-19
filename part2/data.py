from dataclasses import dataclass
from typing import Dict, List
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import scanpy as sc

from utils import to_numpy_matrix
def safe_label_array(s: pd.Series) -> np.ndarray:
    return s.astype(str).to_numpy()
def preprocess_features(x):
    x = normalize(x, norm="l2").astype(np.float32)
    return x
@dataclass
class SplitData:
    train_x: np.ndarray   #Feature matrix of the training set
    train_y: np.ndarray   #Integer labels of the training set, used directly for model training[0, 2, 1, 0, 2, ...]
    train_labels: np.ndarray          # String labels of the training set, corresponding one-to-one with train_y, ['Astrocyte', 'T cell', 'B cell', ...]
    test_x: np.ndarray    #Feature matrix of the test set
    test_y: np.ndarray    #Integer labels of the test set  [1, 0, 2, 1, ...]
    label_map: Dict[int, str]         # Integer -> string, used to decode model outputs  {0: 'Astrocyte', 1: 'B cell', 2: 'T cell'}
    int_map: Dict[str, int]           # String -> integer, used for encoding (inverse of label_map) {'Astrocyte': 0, 'B cell': 1, 'T cell': 2}
    valid_labels: List[str]          #Ordered list of all valid labels; also determines the number of classes in the model output layer  ['Astrocyte', 'B cell', 'T cell']
    train_adata: object             #Filtered original training-set object	Retains .obs, .var, and other metadata for later visualization, UMAP, etc.
    test_adata: object              #Filtered original test-set object	Same as above; predictions can also be written back to an .obs column


def build_split_data(args, logger):

    logger.info("=" * 60)
    logger.info("Data construction started")
    logger.info("=" * 60)

    adata = sc.read_h5ad(args.input_h5ad)
    batch = adata.obs['Batch'].astype(str)
    train_ad = adata[batch == str(args.source_batch)].copy()
    test_ad  = adata[batch == str(args.target_batch)].copy()

    logger.info(f"Input file: {args.input_h5ad}")
    logger.info(f"Training batch: {args.source_batch}  |  Test batch: {args.target_batch}")
    logger.info(f"Input training cells: {train_ad.n_obs}  |  Input test cells: {test_ad.n_obs}")
    logger.info(f"Input feature dimension: {train_ad.n_vars}")

    # ---- Step 0: Filter Unknown ----
    logger.info("-" * 60)
    logger.info(f"Filter Unknown: {args.del_unknown}")
    if args.del_unknown:
        train_ad = train_ad[train_ad.obs['CellType'] != 'Unknown'].copy()
        train_ad = train_ad[train_ad.obs['CellType'] != 'unknown'].copy()
        train_ad = train_ad[train_ad.obs['CellType'] != 'UNK'].copy()

        test_ad  = test_ad[test_ad.obs['CellType'] != 'Unknown'].copy()
        test_ad  = test_ad[test_ad.obs['CellType'] != 'unknown'].copy()
        test_ad = test_ad[test_ad.obs['CellType'] != 'UNK'].copy()
        logger.info(f"Training cells after filtering Unknown: {train_ad.n_obs}  |  Test cells: {test_ad.n_obs}")

    # ---- Step 1~4: Filter rare classes + take intersection ----
    logger.info("-" * 60)
    logger.info(f"Filter rare classes (<=10): {args.del_rare}")
    if args.del_rare:
        label_name = 'CellType'

        # Training set: remove rare classes with <=10 cells
        class_list = np.unique(train_ad.obs[label_name].values)
        rare_train = [c for c in class_list if sum(train_ad.obs[label_name].values == c) <= 10]
        for c in rare_train:
            train_ad = train_ad[train_ad.obs[label_name] != c]
        if rare_train:
            logger.info(f"Rare classes removed from the training set (<=10): {rare_train}")

        # Test set: remove rare classes with <=10 cells
        class_list = np.unique(test_ad.obs[label_name].values)
        rare_test = [c for c in class_list if sum(test_ad.obs[label_name].values == c) <= 10]
        for c in rare_test:
            test_ad = test_ad[test_ad.obs[label_name] != c]
        if rare_test:
            logger.info(f"Rare classes removed from the test set (<=10): {rare_test}")

        # Take the intersection and remove test-set labels absent from the training set
        intersection = np.intersect1d(
            np.unique(train_ad.obs[label_name].values),
            np.unique(test_ad.obs[label_name].values)
        )
        difference = np.setdiff1d(np.unique(test_ad.obs[label_name].values), intersection)
        for c in difference:
            test_ad = test_ad[test_ad.obs[label_name] != c]

        train_ad = train_ad.copy()
        test_ad  = test_ad.copy()

        logger.info(f"Number of intersection labels: {len(intersection)}")
        if len(difference) > 0:
            logger.info(f"Removed test-only labels: {list(difference)}")
        logger.info(f"Training cells after filtering rare classes: {train_ad.n_obs}  |  Test cells: {test_ad.n_obs}")

    # ---- Feature extraction ----
    train_x = preprocess_features(to_numpy_matrix(train_ad.X))
    test_x  = preprocess_features(to_numpy_matrix(test_ad.X))

    train_labels = safe_label_array(train_ad.obs['CellType'])
    test_labels  = safe_label_array(test_ad.obs['CellType'])

    # ---- Build label mapping ----
    valid_labels = sorted(np.unique(train_labels).tolist())
    label_map = {i: l for i, l in enumerate(valid_labels)}
    int_map   = {l: i for i, l in label_map.items()}

    train_y = np.array([int_map[x] for x in train_labels], dtype=np.int64)
    test_y  = np.array([int_map[x] for x in test_labels],  dtype=np.int64)

    # ---- Label mapping ----
    logger.info("-" * 60)
    logger.info(f"Number of valid labels (classes): {len(valid_labels)}")
    logger.info(f"Feature dimension: {train_x.shape[1]}D  |  Preprocessing: L2 normalize")
    logger.info("-" * 60)
    logger.info("Label mapping (label -> integer encoding):")
    for label, idx in int_map.items():
        train_count = np.sum(train_labels == label)
        test_count  = np.sum(test_labels == label)
        logger.info(f"  {idx:3d} ← {label:<30s}  (train: {train_count}, test: {test_count})")

    # ---- Final statistics ----
    logger.info("-" * 60)
    logger.info(f"Final training set: {train_x.shape[0]} cells × {train_x.shape[1]} features, {len(valid_labels)} classes")
    logger.info(f"Final test set: {test_x.shape[0]} cells × {test_x.shape[1]} features")
    logger.info("=" * 60)

    return SplitData(
        train_x=train_x,
        train_y=train_y,
        train_labels=train_labels,
        test_x=test_x,
        test_y=test_y,
        label_map=label_map,
        int_map=int_map,
        valid_labels=valid_labels,
        train_adata=train_ad,
        test_adata=test_ad,
    )
