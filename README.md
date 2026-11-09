# FreqBlender for Kaggle + proposed efficient student

**Current notebook input:** `/kaggle/input/split-ffpp` (CHANGE ACCORDING TO THE DATASET), containing `train/real/*.png`, `train/fake/*.png`, `val/real/*.png`, `val/fake/*.png`, `test/real/*.png`, and `test/fake/*.png`. The four generated notebooks now use Kaggle setup and this PNG layout only. Each builds its own manifest in `/kaggle/working/freqblender/prepared/`, preserving the existing splits and reading images in place. Adjust `FFPP_ROOT` in the setup cell if the mounted path differs.

Upload the regenerated notebooks to Kaggle and attach the same image dataset to each. Enable GPU for training and Internet for dependencies/weights. Save outputs between sessions; attach the preceding output and set `PREVIOUS_RUNS` to its `freqblender/runs` directory to restore checkpoints.

## OBJECTIVE
An independent PyTorch implementation of the FreqBlender paper's method, considering the [**FaceForensics++**](https://www.kaggle.com/datasets/delno002/split-ffpp) dataset , and a proposed **structural secant distillation** step for compact detector inference.

**Workflow Start:** entrypoint - [00_prepare_ffpp.ipynb](notebooks/00_prepare_ffpp.ipynb) in Google Colab (File → Upload notebook), select a GPU runtime for training, and run notebooks in order. The flow expects to be run on kaggle only to prevent dataset downloading and files linking.  

| Notebook | Purpose |
|---|---|
| [00_prepare_ffpp.ipynb](notebooks/00_prepare_ffpp.ipynb) | Mount Drive, obtain official split JSONs, crop FF++ videos, audit splits and preview crops |
| [01_train_freqblender.ipynb](notebooks/01_train_freqblender.ipynb) | Train the auxiliary ResNet34, FPNet, EfficientNet-B4 teacher and optional spatial-only baseline |
| [02_train_efficient_student.ipynb](notebooks/02_train_efficient_student.ipynb) | Train compact CE, ordinary KD, structural-only and full proposed students |
| [03_evaluate_and_export.ipynb](notebooks/03_evaluate_and_export.ipynb) | Held-out evaluation, confidence intervals, matched latency measurements, plots, TorchScript export and crop inference |

The scripts also run locally with a compatible PyTorch/torchvision installation plus `pip install -r requirements.txt`. Colab keeps its own matched CUDA torch/torchvision versions. The FaceNet substitute is installed separately with `--no-deps` to avoid replacing Colab's CUDA packages; notebook 01 checks its import and downloads its pretrained weights on first use.

The paper's augmentation is training-only. The proposed speedup comes from a **smaller deployed detector**; the new loss tries to preserve structural forgery sensitivity during distillation. No faster inference or better AUC can be claimed as measured. See [the mathematical proposal and reproduction differences](report.pdf).

## FaceForensics++ data

Obtain access and download videos using the [official dataset instructions](https://www.kaggle.com/datasets/delno002/split-ffpp). Put the downloaded tree on Drive:
=
## Checkpoints and reproducibility

Each stage writes `best.pt`, `last.pt`, `config.json`, `history.json` to Drive. `--resume` restores model, Adam, AMP scaler, RNG and epoch-level data ordering. Resuming occurs at epoch boundaries; a Colab disconnect mid-epoch repeats that epoch. Config and manifest hashes are checked. Pretrained torchvision/FaceNet weights require internet on first use. Record notebook `pip freeze` outputs with results; package ranges support Colab, they are not a historical environment lock.

The full method trains several networks and may exceed a single free Colab session. Start at 224px, batch 4–8; lower the real-image batch to 2 if needed (the proposed student processes six times that many images per combined forward). Keep image size fixed across compared detectors. Save each stage independently. The notebooks do not pretend short smoke runs reproduce a 200-epoch research experiment.

To validate without a dataset or pretrained downloads:

```bash
python -m pytest tests -q
python scripts/build_notebooks.py
```

Tests cover DCT against SciPy, gradients through frozen auxiliaries, mask/blending semantics, bounded interventions, manifest leakage, checkpoint resume, real model export, and miniature training stages. These verify execution idea. Check `VALIDATION.md` for checks actually run in this workspace.

## Source map

- `freqblender/frequency.py`: DCT, FPNet and frequency blending.
- `freqblender/losses.py`: paper objectives, KD and proposed structural secants.
- `freqblender/preprocess.py`, `data.py`: FF++ crops, official splits and input validation.
- `freqblender/train.py`: staged training, validation, checkpoints and resume.
- `freqblender/evaluate.py`: frame/video metrics, source-group bootstrap, timing and export.
- `scripts/build_notebooks.py`: reproducible notebook generator with embedded source.

Research provenance: [FreqBlender paper](https://arxiv.org/abs/2404.13872), [torchvision EfficientNet-B4](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b4.html), [torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html). All extensions and implementation choices are separated in `report.pdf`.
