# FreqBlender for Kaggle + proposed efficient student

**Current notebook input:** `/kaggle/input/split-ffpp`, containing `train/real/*.png`, `train/fake/*.png`, `val/real/*.png`, `val/fake/*.png`, `test/real/*.png`, and `test/fake/*.png`. The four generated notebooks now use Kaggle setup and this PNG layout only. Each builds its own manifest in `/kaggle/working/freqblender/prepared/`, preserving the existing splits and reading images in place. No raw-video extraction, face detection, Google Drive mount, or split-list download is needed. Adjust `FFPP_ROOT` in the setup cell if the mounted path differs.

Upload the regenerated notebooks to Kaggle and attach the same image dataset to each. Enable GPU for training and Internet for dependencies/weights. Save outputs between sessions; attach the preceding output and set `PREVIOUS_RUNS` to its `freqblender/runs` directory to restore checkpoints. Rebuilding with `scripts/build_notebooks.py` preserves this Kaggle setup.

The PNG layout does not provide source-video identities. Each image receives a compatibility ID for the existing loader; video-named internal metric fields therefore represent **image-level** results here. Source-video leakage and video-level accuracy cannot be established without additional metadata. The raw-video/Colab descriptions below document the earlier workflow; the standalone raw-video preprocessing module remains available, but is not called by these notebooks.

An independent PyTorch implementation of the FreqBlender core method, tailored to **FaceForensics++**, with four self-contained Colab notebooks and a proposed **structural secant distillation** objective for compact detector inference.

**Start:** open [00_prepare_ffpp.ipynb](notebooks/00_prepare_ffpp.ipynb) in Google Colab (File → Upload notebook), select a GPU runtime for training, and run notebooks in order. Each notebook embeds the project source, so no GitHub repository URL or separate source upload is required. Set your Google Drive paths in the configuration cells. FF++ videos and trained weights are not included.

| Notebook | Purpose |
|---|---|
| [00_prepare_ffpp.ipynb](notebooks/00_prepare_ffpp.ipynb) | Mount Drive, obtain official split JSONs, crop FF++ videos, audit splits and preview crops |
| [01_train_freqblender.ipynb](notebooks/01_train_freqblender.ipynb) | Train the auxiliary ResNet34, FPNet, EfficientNet-B4 teacher and optional spatial-only baseline |
| [02_train_efficient_student.ipynb](notebooks/02_train_efficient_student.ipynb) | Train compact CE, ordinary KD, structural-only and full proposed students |
| [03_evaluate_and_export.ipynb](notebooks/03_evaluate_and_export.ipynb) | Held-out evaluation, confidence intervals, matched latency measurements, plots, TorchScript export and crop inference |

The scripts also run locally with a compatible PyTorch/torchvision installation plus `pip install -r requirements.txt`. Colab keeps its own matched CUDA torch/torchvision versions. The FaceNet substitute is installed separately with `--no-deps` to avoid replacing Colab's CUDA packages; notebook 01 checks its import and downloads its pretrained weights on first use.

The paper's augmentation is training-only. The proposed speedup comes from a **smaller deployed detector**; the new loss tries to preserve structural forgery sensitivity during distillation. No faster inference or better AUC has been claimed as measured. See [the mathematical proposal and reproduction differences](docs/METHOD.md).

The baseline implements FPNet's three independent masks, the four objective structures and Eq. 6 blending, but uses documented practical substitutions (FaceNet, simplified spatial self-blending, Haar crops, normalized losses and shorter Colab schedules). It is **not an exact reproduction of the paper's reported results**. Use 400px/200 parser epochs only with sufficient GPU budget; all ablation runs should share their resolution and recipe.

## FaceForensics++ data

Obtain access and download videos using the [official dataset instructions](https://github.com/ondyari/FaceForensics/tree/master/dataset). Put the downloaded tree on Drive:

```text
FaceForensics++/
  original_sequences/youtube/c23/videos/000.mp4
  manipulated_sequences/Deepfakes/c23/videos/000_003.mp4
  manipulated_sequences/Face2Face/c23/videos/000_003.mp4
  manipulated_sequences/FaceSwap/c23/videos/000_003.mp4
  manipulated_sequences/NeuralTextures/c23/videos/000_003.mp4
```

Preprocessing reuses original-video face boxes for the corresponding manipulated videos. Inspect previews and detection failures before training. Haar crops are a portable fallback; high-quality aligned pre-extracted crops can be provided instead through a CSV with:

```csv
path,label,split,video_id,group_id,dataset
crops/original/000/000000.png,0,train,original/000,000_003,FF++
crops/Deepfakes/000_003/000000.png,1,train,Deepfakes/000_003,000_003,FF++
```

Paths are relative to the CSV's parent or absolute; labels are 0=real, 1=fake. `group_id` is the official original pair shared by originals and manipulations; all its frames must stay in one split. Do not create random frame-level splits. The notebook downloads the official train/val/test pair lists. Training uses **only real training crops**, making fake donors on the fly. Real manipulated validation/test crops are required for meaningful AUC.

## Checkpoints and reproducibility

Each stage writes `best.pt`, `last.pt`, `config.json`, `history.json` to Drive. `--resume` restores model, Adam, AMP scaler, RNG and epoch-level data ordering. Resuming occurs at epoch boundaries; a Colab disconnect mid-epoch repeats that epoch. Config and manifest hashes are checked. Pretrained torchvision/FaceNet weights require internet on first use. Record notebook `pip freeze` outputs with results; package ranges support Colab, they are not a historical environment lock.

The full method trains several networks and may exceed a single free Colab session. Start at 224px, batch 4–8; lower the real-image batch to 2 if needed (the proposed student processes six times that many images per combined forward). Keep image size fixed across compared detectors. Save each stage independently. The notebooks do not pretend short smoke runs reproduce a 200-epoch research experiment.

To validate without a dataset or pretrained downloads:

```bash
python -m pytest tests -q
python scripts/build_notebooks.py
```

Tests cover DCT against SciPy, gradients through frozen auxiliaries, mask/blending semantics, bounded interventions, manifest leakage, checkpoint resume, real model export, and miniature training stages. These verify execution/numerics, not deepfake accuracy. See `VALIDATION.md` for checks actually run in this workspace.

## Source map

- `freqblender/frequency.py`: DCT, FPNet and frequency blending.
- `freqblender/losses.py`: paper objectives, KD and proposed structural secants.
- `freqblender/preprocess.py`, `data.py`: FF++ crops, official splits and input validation.
- `freqblender/train.py`: staged training, validation, checkpoints and resume.
- `freqblender/evaluate.py`: frame/video metrics, source-group bootstrap, timing and export.
- `scripts/build_notebooks.py`: reproducible notebook generator with embedded source.

Research provenance: [FreqBlender paper](https://arxiv.org/abs/2404.13872), [torchvision EfficientNet-B4](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b4.html), [torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html). All extensions and implementation choices are separated in `docs/METHOD.md`.
