# ProCAT 123

ProCAT is organized as a three-step pipeline:

1. **Stage 1** learns cell embeddings from the original accessibility matrix.
2. **Split step** assigns cells to `reference` / `query` using the original cell-type labels and transfers those annotations to the Stage-1 embedding by barcode.
3. **Stage 2** trains the cell-type model on `reference` cells and evaluates/predicts `query` cells.

## Repository layout

```text
ProCAT/
├── part1/
│   ├── accessibility_network.py
│   ├── channel_attention.py
│   ├── genomic_dataset.py
│   ├── train_pipeline.py
│   └── training_utils.py
├── scripts/
│   └── splitdata.py
├── part2/
│   ├── data.py
│   ├── main.py
│   ├── model.py
│   └── utils.py
├── environment.yml
├── .gitignore
└── README.md
```

## 1. Create the environment

The Conda environment is named **`ProCAT`**.

```bash
conda env create -f environment.yml
conda activate ProCAT
```

The provided environment uses PyTorch 2.4.1 with CUDA 11.8. If your server requires another CUDA build, replace `pytorch-cuda=11.8` with a PyTorch-supported CUDA version before creating the environment.

Check the installation:

```bash
python -c "import torch, scanpy, anndata, sklearn; print(torch.__version__); print(torch.cuda.is_available())"
```

## 2. Input requirements

The original `.h5ad` used by Stage 1 should contain:

- accessibility matrix in `adata.X`;
- genomic peak coordinates in `adata.var['chr']`, `adata.var['start']`, and `adata.var['end']`;
- cell-type labels in `adata.obs['CellType']` or `adata.obs['cell_type']`;
- preferably `adata.obs['barcode']`; when it is absent, `obs_names` are used as barcodes.

Stage 1 also requires the encoded reference-genome HDF5 file used by the sequence model. Pass it explicitly with `--genome-file`.

## 3. Stage 1

Example for hg38:

```bash
python part1/train_pipeline.py \
  -i /path/to/ad.h5ad \
  -g hg38 \
  --genome-file /path/to/hg38.fa.h5 \
  -o results/part1 \
  --device 0
```

Main output:

```text
results/part1/CACNN_output.h5ad
```

The old machine-specific genome paths have been removed. You can also set a genome path once through an environment variable, for example:

```bash
export PROCAT_GENOME_HG38=/path/to/hg38.fa.h5
python part1/train_pipeline.py -i /path/to/ad.h5ad -g hg38 -o results/part1 --device 0
```

## 4. Split the Stage-1 output

Run this **after Stage 1 and before Stage 2**:

```bash
python scripts/splitdata.py \
  --original-h5ad /path/to/ad.h5ad \
  --embedding-h5ad results/part1/CACNN_output.h5ad \
  --ratio 0.6 \
  --seed 42 \
  --output-dir results/split
```

This creates:

```text
results/split/radio_ad.h5ad
results/split/radio_partOne_embed.h5ad
```

`radio_partOne_embed.h5ad` contains the Stage-1 embedding plus the two columns required by Stage 2:

```text
Batch:    reference / query
CellType: cell-type label
```

## 5. Stage 2

Use the split embedding as the Stage-2 input:

```bash
python part2/main.py \
  -i results/split/radio_partOne_embed.h5ad \
  --source_batch reference \
  --target_batch query \
  --gpu 0 \
  --o results/part2 \
  --o_name run1
```

Main output:

```text
results/part2/run1/embedding.h5ad
```

## 6. Full workflow

```text
original ad.h5ad
      │
      ▼
part1/train_pipeline.py
      │
      ▼
CACNN_output.h5ad
      │
      ├──────── original ad.h5ad
      │                │
      └──── scripts/splitdata.py
                       │
                       ▼
              radio_partOne_embed.h5ad
                       │
                       ▼
                 part2/main.py
                       │
                       ▼
                  embedding.h5ad
```

## 7. Upload to GitHub

Do not commit `.h5ad`, genome `.h5`, trained `.pt`, or generated result files. They are excluded by `.gitignore`.

Create an empty repository named `ProCAT` on GitHub, then run from the local `ProCAT` directory:

```bash
git init
git add .
git commit -m "Initial ProCAT release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ProCAT.git
git push -u origin main
```

For future updates:

```bash
git add .
git commit -m "Update ProCAT"
git push
```
