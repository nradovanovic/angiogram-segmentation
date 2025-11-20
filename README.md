# Coronary Artery Segmentation in X-ray Angiograms

Automated segmentation of coronary arteries from X-ray angiograms using deep learning. This project provides preprocessing utilities, multiple segmentation architectures, a reproducible training pipeline, and evaluation tools to compare model performance on a dataset of 134 angiograms and corresponding vessel masks.

## Environment Setup

1. **Python version**: 3.9+ is recommended.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Verify CUDA availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

## Dataset

The dataset ships with the repository under `Database_134_Angiograms/`. Each angiogram (`*.pgm`) has a corresponding binary mask (`*_gt.pgm`). If you relocate the data, update the `--data_dir` argument when running training scripts.

## Usage

### Preprocessing & Visualisation

You can explore the data either interactively via `notebooks/angiogram_eda.ipynb`, via the command-line tools, or programmatically with the helpers in `src.preprocessing`.

#### CLI workflow

- **Explore dataset & save overlays**
  ```bash
  python -m src.explore_dataset \
    --data_dir Database_134_Angiograms \
    --output_dir results/eda \
    --num_samples 6 \
    --no_show
  ```
  Produces `dataset_stats.json` and `sample_overlays.png` under `results/eda/`.

- **Generate reproducible splits and export arrays**
  ```bash
  python -m src.preprocess_dataset \
    --data_dir Database_134_Angiograms \
    --output_dir results/preprocessed \
    --image_size 512 512 \
    --val_size 0.15 \
    --test_size 0.15 \
    --export \
    --format npy
  ```
  Creates `splits.json` and (when `--export` is set) saves resized `.npy` tensors plus a `manifest.json` describing the exported files. Omit `--export` to record only the split definitions.

#### Manual workflow

**1. Inspect basic statistics**

```python
from pathlib import Path
from src.preprocessing import load_image_mask_pairs, describe_dataset

pairs = load_image_mask_pairs(Path("Database_134_Angiograms"))
stats = describe_dataset(pairs)
print(f"Samples: {stats['num_samples']}")
print(f"Image mean/std: {stats['image_mean']:.2f} / {stats['image_std']:.2f}")
print(f"Average mask coverage: {stats['mask_mean'] * 100:.2f}%")
```

**2. Create train/val/test datasets with augmentations**

```python
from src.preprocessing import create_datasets

train_ds, val_ds, test_ds = create_datasets(
    data_dir="Database_134_Angiograms",
    image_size=(512, 512),   # change to your target resolution
    val_size=0.15,
    test_size=0.15,
    seed=42,
    augment=True,            # set False to disable augmentation on the train split
)
```

If you need batched tensors for prototyping, wrap the returned datasets with `create_dataloaders`.

**3. Preview raw and augmented samples**

```python
from src.preprocessing import visualize_samples

# Show raw, resized samples from the validation split
visualize_samples(val_ds, num_samples=4)

# To look at augmented variants, draw directly from the training dataset
visualize_samples(train_ds, num_samples=4)
```

`visualize_samples` overlays the binary mask on top of the grayscale angiogram. Use the `figsize` argument to control plot layout when exporting figures.

**4. Sanity-check custom augmentations**

The augmentation pipeline is defined in `src.data.dataset.default_augmentation_pipeline`. Modify it to include domain-specific transforms (e.g., motion blur) and re-run the visualisation snippet above to confirm the masks remain aligned.

### Training

Train one or multiple models with reproducible splits and augmentations:

```bash
python -m src.train \
  --data_dir Database_134_Angiograms \
  --output_dir results \
  --models unetpp unet3plus transunet \
  --epochs 100 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --amp
```

Key options:

- `--models`: any subset of `unet`, `unetpp`, `unet3plus`, `transunet`
- `--loss`: `dice`, `bce_dice`, or `focal`
- `--image_size`: target resize `(height width)`
- `--patience`: early stopping patience
- `--amp`: enable automatic mixed precision on CUDA

Checkpoints and per-epoch metric logs are stored in `results/checkpoints/` and `results/metrics/`.

### Training on Kaggle GPU

Use this workflow to run the exact same pipeline inside a Kaggle Notebook with a free GPU.

1. **Start a Kaggle Notebook**
   - Pick *GPU → T4/P100*, turn Internet ON (needed if you pull from git), and attach the angiogram dataset so `/kaggle/input/.../Database_134_Angiograms` exists.
   - Easiest path: upload `notebooks/kaggle_full_pipeline.ipynb`, open it on Kaggle, and run the cells sequentially. The notebook recreates the repo in `/kaggle/working/angiogram-segmentation`, installs dependencies, preprocesses data, trains all three models, and exports plots/metrics automatically.
2. **Manual workflow (if you prefer shell commands)**
   - Clone or copy the repo into `/kaggle/working/angiogram-segmentation`.
   - Install dependencies:
     ```bash
     %cd /kaggle/working/angiogram-segmentation
     pip install -r requirements.txt
     ```
   - (Optional) Run preprocessing:
     ```bash
     python -m src.preprocess_dataset \
       --data_dir Database_134_Angiograms \
       --output_dir results/preprocessed \
       --image_size 512 512 \
       --export --format npy
     ```
   - Train on GPU:
     ```bash
     python -m src.train \
       --output_dir /kaggle/working/results \
       --models unetpp unet3plus transunet \
       --epochs 100 \
       --batch_size 8 \
       --num_workers 2 \
       --amp
     ```
     `src.train` auto-detects attached Kaggle datasets under `/kaggle/input/**/Database_134_Angiograms`, so you only need to override `--data_dir` if the folder name differs.
   - When training finishes, click *Save Version* so `/kaggle/working/results/**/*` (checkpoints + metrics) are stored with the Notebook version. Optionally zip the `results/` folder and publish it as a dataset for download.

Tips:

- Keep `--num_workers` between 2–4 to stay within the Notebook’s RAM budget.
- Always write outputs to `/kaggle/working` (`/kaggle/input` is read-only).
- Prefer the ready-made Kaggle notebook at `notebooks/kaggle_full_pipeline.ipynb` if you want a single file that rebuilds the repo, preprocesses data, trains UNet++, UNet 3+, and TransUNet, and generates plots with no extra setup.

### Evaluation & Comparison

After training, aggregate test metrics and generate comparison plots:

```bash
python -m src.plot_results \
  --metrics_dir results/metrics \
  --output_dir results/plots
```

This produces a `metrics_summary.csv` table and bar plots (`dice`, `iou`, `precision`, `recall`) under `results/plots/`.

### Testing on the Held-out Set

`src/train.py` automatically evaluates the best checkpoint (by validation Dice) on the test split and stores metrics in `results/metrics/<model>_metrics.json`. To run inference on individual images, load the checkpoint:

```python
import torch
from src.models import UNetPlusPlus

model = UNetPlusPlus()
state = torch.load("results/checkpoints/unetpp_best.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

## Reproducibility Notes

- All scripts call `seed_everything` to fix seeds across `random`, `numpy`, and `torch`.
- Data splits are deterministic given the `--seed` argument.
- Augmentations use Albumentations (`horizontal/vertical flips`, `random rotations`, `contrast`, `elastic distortions`, and `noise`).
- Mixed precision (`--amp`) provides additional reproducibility safeguards via deterministic cuDNN settings when seeds are fixed.


