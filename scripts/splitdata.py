"""Split Stage-1 output into reference/query data for Stage 2.

The split is generated on the original AnnData object and then mapped to the
Stage-1 embedding by barcode. This keeps Stage 1 independent of the
reference/query split while giving Stage 2 the required ``Batch`` and
``CellType`` columns.
"""

import argparse
import os

import numpy as np
import scanpy as sc
from scipy import sparse


def ensure_dense(ad):
    ad = ad.copy()
    if sparse.issparse(ad.X):
        ad.X = ad.X.toarray().astype(np.float32)
    else:
        ad.X = np.asarray(ad.X, dtype=np.float32)
    return ad


def del_ratio(ad, ratio, min_ref, seed):
    ad = ad.copy()

    if "CellType" not in ad.obs.columns:
        if "cell_type" in ad.obs.columns:
            ad.obs["CellType"] = ad.obs["cell_type"].astype(str).copy()
            print("Created CellType column from cell_type")
        else:
            raise ValueError("Input ad.h5ad is missing CellType / cell_type column")
    else:
        ad.obs["CellType"] = ad.obs["CellType"].astype(str)

    if "barcode" not in ad.obs.columns:
        ad.obs["barcode"] = ad.obs_names.astype(str)
        print("No barcode found in original data; created barcode from obs_names")
    else:
        ad.obs["barcode"] = ad.obs["barcode"].astype(str)

    if ad.obs["barcode"].duplicated().any():
        raise ValueError("Duplicate barcodes found in the original data")

    if "Batch" in ad.obs.columns:
        ad.obs.drop(columns=["Batch"], inplace=True)
        print("Removed existing Batch column; a new Batch column will be generated")

    ad.obs["Batch"] = "query"

    rng = np.random.default_rng(seed)
    for cell_type in ad.obs["CellType"].unique():
        indices = ad.obs.index[ad.obs["CellType"] == cell_type]
        n_total = len(indices)
        n_ref = int(n_total * ratio)

        if n_ref <= 1:
            n_ref = min(min_ref, n_total)

        ref_indices = rng.choice(indices, size=n_ref, replace=False)
        ad.obs.loc[ref_indices, "Batch"] = "reference"

    return ad


def copy_obs_by_barcode(ad_source, ad_target):
    ad_source = ad_source.copy()
    ad_target = ad_target.copy()

    if "barcode" not in ad_source.obs.columns:
        ad_source.obs["barcode"] = ad_source.obs_names.astype(str)
    else:
        ad_source.obs["barcode"] = ad_source.obs["barcode"].astype(str)

    if "barcode" not in ad_target.obs.columns:
        ad_target.obs["barcode"] = ad_target.obs_names.astype(str)
        print("No barcode found in Stage-1 embedding; created barcode from obs_names")
    else:
        ad_target.obs["barcode"] = ad_target.obs["barcode"].astype(str)

    if ad_source.obs["barcode"].duplicated().any():
        raise ValueError("Duplicate barcodes found in ad_source; cannot map safely")
    if ad_target.obs["barcode"].duplicated().any():
        raise ValueError("Duplicate barcodes found in ad_target; cannot map safely")

    map_df = (
        ad_source.obs[["barcode", "Batch", "CellType"]]
        .drop_duplicates("barcode")
        .set_index("barcode")
    )

    if "Batch" in ad_target.obs.columns:
        ad_target.obs.drop(columns=["Batch"], inplace=True)
    ad_target.obs["Batch"] = (
        ad_target.obs["barcode"].map(map_df["Batch"]).fillna("query").astype(str)
    )

    if "CellType" in ad_target.obs.columns:
        ad_target.obs.drop(columns=["CellType"], inplace=True)
    ad_target.obs["CellType"] = (
        ad_target.obs["barcode"].map(map_df["CellType"]).fillna("Unknown").astype(str)
    )

    n_matched = ad_target.obs["barcode"].isin(ad_source.obs["barcode"]).sum()
    n_total = len(ad_target.obs)
    print(f"Barcode matched: {n_matched}/{n_total}")
    if n_matched != n_total:
        print("Warning: unmatched Stage-1 cells were assigned Batch=query and CellType=Unknown")

    return ad_target


def check_for_training(ad):
    print("========== Checking output file ==========")
    print(ad)
    print("X type:", type(ad.X))
    print("X shape:", ad.X.shape)
    print("X dtype:", ad.X.dtype)
    print("Batch counts:")
    print(ad.obs["Batch"].value_counts(dropna=False))
    print("CellType missing:", ad.obs["CellType"].isna().sum())

    assert "Batch" in ad.obs.columns
    assert "CellType" in ad.obs.columns
    assert (ad.obs["Batch"] == "reference").sum() > 0
    assert (ad.obs["Batch"] == "query").sum() > 0
    assert not ad.obs["CellType"].isna().any()
    assert isinstance(ad.X, np.ndarray)

    if np.isnan(ad.X).any():
        raise ValueError("NaN values found in X")
    if np.isinf(ad.X).any():
        raise ValueError("Inf values found in X")

    print("Check passed. The data can be directly used by Stage 2.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Split original data into reference/query and transfer the split to Stage-1 embeddings."
    )
    parser.add_argument(
        "--original-h5ad",
        required=True,
        help="Original input h5ad used by Stage 1.",
    )
    parser.add_argument(
        "--embedding-h5ad",
        required=True,
        help="Stage-1 CACNN_output.h5ad file.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.6,
        help="Fraction of each cell type assigned to the reference set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-ref", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the Stage-1 embedding directory.",
    )
    parser.add_argument(
        "--original-output-name",
        default="radio_ad.h5ad",
        help="Filename for the split original AnnData.",
    )
    parser.add_argument(
        "--embedding-output-name",
        default="radio_partOne_embed.h5ad",
        help="Filename for the split Stage-1 embedding used by Stage 2.",
    )
    return parser


def main(args):
    if not (0 < args.ratio < 1):
        raise ValueError("--ratio must be strictly between 0 and 1")
    if args.min_ref < 1:
        raise ValueError("--min-ref must be >= 1")
    if not os.path.isfile(args.original_h5ad):
        raise FileNotFoundError(args.original_h5ad)
    if not os.path.isfile(args.embedding_h5ad):
        raise FileNotFoundError(args.embedding_h5ad)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.embedding_h5ad))
    os.makedirs(output_dir, exist_ok=True)

    original = sc.read_h5ad(args.original_h5ad)
    original = del_ratio(original, args.ratio, args.min_ref, args.seed)

    original_dense = ensure_dense(original)
    original_out = os.path.join(output_dir, args.original_output_name)
    original_dense.write_h5ad(original_out, compression="gzip")
    print(f"Saved split original data: {original_out}")

    embedding = sc.read_h5ad(args.embedding_h5ad)
    embedding = copy_obs_by_barcode(original, embedding)
    embedding = ensure_dense(embedding)
    check_for_training(embedding)

    embedding_out = os.path.join(output_dir, args.embedding_output_name)
    embedding.write_h5ad(embedding_out, compression="gzip")
    print(f"Saved Stage-2 input: {embedding_out}")


if __name__ == "__main__":
    main(build_parser().parse_args())
