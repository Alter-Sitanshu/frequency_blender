"""Build portable Kaggle notebooks containing a compressed snapshot of source."""
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip()+"\n"}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": textwrap.dedent(source).strip()+"\n"}


def snapshot():
    buffer = io.BytesIO()
    paths = sorted(list((ROOT / "freqblender").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
                   + [ROOT / "requirements.txt", ROOT / "docs/METHOD.md", ROOT / "README.md",
                      ROOT / "VALIDATION.md", ROOT / "scripts/build_notebooks.py"])
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    data = buffer.getvalue()
    return base64.b64encode(data).decode(), hashlib.sha256(data).hexdigest()


def common():
    payload, digest = snapshot()
    return [md("""
        ## Runtime and embedded project setup
        For training, select **Runtime → Change runtime type → GPU**. This cell unpacks
        the complete project source into `/kaggle/working/freqblender_project` and installs support
        dependencies. It keeps Kaggle's matched PyTorch/torchvision CUDA installation.
        No GitHub fork or separate project ZIP upload is needed.
        """), code(f'''
        import base64, hashlib, io, json, os, sys, subprocess, zipfile
        from pathlib import Path
        payload = base64.b64decode("{payload}")
        assert hashlib.sha256(payload).hexdigest() == "{digest}"
        PROJECT = Path("/kaggle/working/freqblender_project")
        PROJECT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                target = (PROJECT / member.filename).resolve()
                if not target.is_relative_to(PROJECT.resolve()):
                    raise ValueError("Unsafe archive path")
            archive.extractall(PROJECT)
        os.chdir(PROJECT)
        if str(PROJECT) not in sys.path:
            sys.path.insert(0, str(PROJECT))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
        import torch, torchvision
        print("PyTorch", torch.__version__, "torchvision", torchvision.__version__)
        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        '''), md("""
        ## Kaggle input and writable outputs
        Attach `delno002/split-ffpp` to every notebook. Its root must contain
        train/real, train/fake, val/real, val/fake, test/real and test/fake PNG folders.
        Existing splits are preserved. Images are read directly from the attached dataset.
        Save notebook outputs before ending a session. For later stages, attach the previous
        saved output and set PREVIOUS_RUNS to its `freqblender/runs` directory.
        """), code('''
        import shutil
        FFPP_ROOT = Path("/kaggle/input/split-ffpp")  # EDIT if Kaggle mounts a different path
        DRIVE_PROJECT = Path("/kaggle/working/freqblender")  # legacy variable name, local Kaggle path
        DATA_ROOT = DRIVE_PROJECT / "prepared"
        RUN_ROOT = DRIVE_PROJECT / "runs"
        MANIFEST = DATA_ROOT / "manifest.csv"
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        PREVIOUS_RUNS = None  # e.g. Path("/kaggle/input/previous-output/freqblender/runs")
        if PREVIOUS_RUNS is not None:
            # Copy missing files only, preserving newer checkpoints from this session.
            previous = Path(PREVIOUS_RUNS)
            if not previous.is_dir():
                raise FileNotFoundError(previous)
            for source in previous.rglob("*"):
                target = RUN_ROOT / source.relative_to(previous)
                if source.is_file() and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        from freqblender.data import build_image_manifest
        MANIFEST = build_image_manifest(FFPP_ROOT, MANIFEST)
        versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (DRIVE_PROJECT / "environment.txt").write_text(versions)
        print("Manifest:", MANIFEST)
        print("Using image-level samples; original video/source identities are unavailable.")
        ''')]


def training_helpers():
    return [code('''
        assert torch.cuda.is_available(), "Choose a GPU runtime before training."
        from freqblender.data import read_manifest
        table = read_manifest(MANIFEST)
        display(table.groupby(["split", "label"]).size().unstack(fill_value=0))
        SIZE = 224          # 400 is supported; use SAME size in all stages/ablations
        BATCH_SIZE = 4      # real-image batch; proposed stage processes 6x this per forward
        WORKERS = 2
        SEED = 42
        MAX_STEPS = 0       # 0 = full epochs; positive values are execution tests only
        AMP = False         # FP32 recommended for small secant response differences
        def train_stage(stage, name, epochs, architecture="efficientnet_b4", extra=()):
            command = [sys.executable, "-m", "freqblender.train", "--stage", stage,
                       "--manifest", str(MANIFEST), "--output", str(RUN_ROOT/name),
                       "--architecture", architecture, "--size", str(SIZE),
                       "--batch-size", str(BATCH_SIZE), "--workers", str(WORKERS),
                       "--epochs", str(epochs), "--seed", str(SEED),
                       "--max-steps", str(MAX_STEPS), "--resume"]
            if AMP:
                command += ["--amp"]
            subprocess.run(command + list(map(str, extra)), check=True)
            return RUN_ROOT/name/"best.pt"
        ''')]


def prepare():
    return [md("""
        # 00 ? Prepare split FF++ PNG images on Kaggle
        Expected layout: `train/{real,fake}/*.png`, `val/{real,fake}/*.png`,
        `test/{real,fake}/*.png`. The setup indexes the existing PNG files directly;
        it does not extract videos, detect faces, download split lists, or re-split images.
        Images should already be suitable face crops; inspect the preview below.
        """)] + common() + [md("""
        ## Inspect the manifest
        Labels: real=0, fake=1. Each PNG is one sample. `video_id` and `group_id`
        contain image identifiers solely for compatibility with the training code.
        They are not recovered video identities. Consequently evaluation is image-level,
        and source-video leakage cannot be ruled out from this folder layout alone.
        """), code('''
        from freqblender.data import read_manifest
        table = read_manifest(MANIFEST)
        display(table.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns={0: "real", 1: "fake"}))
        display(table.head())
        '''), code('''
        import matplotlib.pyplot as plt
        from PIL import Image
        preview = table[table.split == "train"].sample(min(12, len(table[table.split == "train"])), random_state=42)
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for axis in axes.flat:
            axis.axis("off")
        for axis, (_, row) in zip(axes.flat, preview.iterrows()):
            with Image.open(row.path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{Path(row.path).name} | label={row.label}", fontsize=8)
        plt.tight_layout()
        '''), md("""
        Continue with **01_train_freqblender.ipynb** and attach the same PNG dataset.
        Each notebook recreates the manifest without copying images. Save trained `runs/`
        outputs between sessions and restore them using PREVIOUS_RUNS in the setup cell.
        """)]


def baseline():
    cells = [md("""
        # 01 · Train FreqBlender and the detector teacher
        Stages: auxiliary ResNet34 → FPNet → EfficientNet-B4 detector.
        Only real **training** frames are used; fake training images are generated on the fly.
        Real/fake validation images select detector checkpoints. The test split is not used here.

        This is a paper-based implementation with explicit adaptations: approximate spatial
        self-blending, supplied PNG crops, FaceNet identity features and normalized losses.
        The default short schedule is a starting experiment, not a reproduction of published AUC.
        See the embedded `docs/METHOD.md` for every difference.
        """)] + common() + training_helpers()
    cells += [code('''
        WARMUP_EPOCHS = 10
        PARSER_EPOCHS = 30  # paper: 200 at 400px; budget/configure explicitly
        TEACHER_EPOCHS = 30
        WARMUP = RUN_ROOT / "warmup" / "best.pt"
        PARSER = RUN_ROOT / "parser" / "best.pt"
        TEACHER = RUN_ROOT / "teacher_freqblender" / "best.pt"
        '''), md("""
        ## 1. Train the authenticity auxiliary
        This ResNet34 learns real versus self-blended faces. FPNet later freezes its weights
        while preserving the gradient from its predictions to reconstructed images.
        """), code('''
        WARMUP = train_stage("warmup", "warmup", WARMUP_EPOCHS, "resnet34")
        '''), md("""
        ## 2. Prepare the frozen face-identity encoder
        Default: FaceNet InceptionResnetV1 pretrained on VGGFace2, a **substitute** for the
        paper's MobileNet-ArcFace. Its weights download automatically on first use.
        To use a compatible MobileNet-ArcFace TorchScript file, set `IDENTITY_CHECKPOINT`.
        Contract: RGB [-1,1] at `IDENTITY_SIZE` -> [batch, embedding dimension]. The wrapper
        normalizes embeddings. Do not use random identity weights or video IDs as identities.
        """), code('''
        IDENTITY_CHECKPOINT = None  # e.g. DRIVE_PROJECT / "weights/mobilenet_arcface.ts"
        IDENTITY_SIZE = 112
        if IDENTITY_CHECKPOINT is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "facenet-pytorch==2.6.0"], check=True)
            from facenet_pytorch import InceptionResnetV1
        from freqblender.models import IdentityEncoder
        identity_check = IdentityEncoder(IDENTITY_CHECKPOINT, IDENTITY_SIZE).eval()
        probe = torch.rand(2, 3, 160, 160, requires_grad=True)
        embedding = identity_check(probe)
        embedding[:, 0].sum().backward()
        assert probe.grad is not None and torch.isfinite(probe.grad).all()
        print(identity_check.kind, tuple(embedding.shape), "input gradients OK")
        del identity_check, probe, embedding
        (DRIVE_PROJECT / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        '''), code('''
        identity_args = [] if IDENTITY_CHECKPOINT is None else ["--identity-checkpoint", IDENTITY_CHECKPOINT,
                                                               "--identity-size", IDENTITY_SIZE]
        PARSER = train_stage("parser", "parser", PARSER_EPOCHS,
                             extra=["--discriminator", WARMUP] + identity_args)
        '''), md("""
        ## Inspect frequency masks and synthesized images
        Mask sums are softly regularized, not forced to one in the paper baseline.
        Saturated/constant masks or badly distorted images indicate FPNet needs investigation
        before spending compute on teacher/student training.
        """), code('''
        import matplotlib.pyplot as plt
        from freqblender.train import load_parser
        from freqblender.data import make_loader
        from freqblender.augment import spatial_self_blend
        parser = load_parser(PARSER, "cuda")
        batch = next(iter(make_loader(table[(table.split == "train") & (table.label == 0)], SIZE, 2, workers=0)))
        real = batch["image"].cuda()
        with torch.no_grad():
            fake = spatial_self_blend(real)
            _, masks = parser(fake)
            blended = parser.blend(real, fake)
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        for axis, tensor, name in zip(axes[0], [real[0], fake[0], blended[0]], ["real", "spatial fake", "frequency blend"]):
            axis.imshow(tensor.cpu().permute(1, 2, 0).clamp(0, 1)); axis.set_title(name); axis.axis("off")
        for axis, mask, name in zip(axes[1], masks[0], ["semantic", "structural", "noise"]):
            axis.imshow(mask.cpu(), vmin=0, vmax=1); axis.set_title(name); axis.axis("off")
        plt.tight_layout()
        print("Mean mask integrity error:", (masks.sum(1)-1).abs().mean().item())
        del parser, real, fake, blended, masks
        torch.cuda.empty_cache()
        '''), md("""
        ## 3. Train the FreqBlender teacher
        FPNet is frozen. Each self-blended fake receives frequency blending with probability 0.2.
        All detector inputs use the same RGB range and built-in ImageNet normalization.
        """), code('''
        TEACHER = train_stage("teacher", "teacher_freqblender", TEACHER_EPOCHS,
                              extra=["--parser", PARSER])
        '''), md("""
        ## Optional but recommended: spatial-only baseline
        This isolates the benefit of frequency augmentation within this implementation.
        It uses the same B4 architecture, initialization seed, schedule and resolution.
        """), code('''
        RUN_SPATIAL_BASELINE = True
        if RUN_SPATIAL_BASELINE:
            train_stage("teacher", "teacher_spatial", TEACHER_EPOCHS)
        '''), code('''
        import pandas as pd
        for name in ("warmup", "parser", "teacher_freqblender", "teacher_spatial"):
            history = RUN_ROOT / name / "history.json"
            if history.exists():
                display(pd.DataFrame(json.loads(history.read_text())).tail().assign(run=name))
        '''), md("""
        Checkpoints are saved after every epoch. Rerun the same cells to resume; changing
        experiment-defining settings requires a new run directory. Open **02_train_efficient_student.ipynb** next.
        """)]
    cells.insert(1, md("This PNG dataset has no supplied video/source grouping. All video-named "
                       "metrics, validation scores, thresholds and bootstrap groups below operate "
                       "on individual images. They must be reported as image-level results, "
                       "not source-video results. Existing folder splits are preserved."))
    return cells


def student():
    cells = [md(r"""
        # 02 · Efficient student and proposed structural secant distillation
        The teacher is EfficientNet-B4; the default deployed student is MobileNetV3-Small.
        FreqBlender already has no inference-time parser, so acceleration comes from the compact architecture.

        Proposed extension: transfer the teacher's local logit response to FPNet-defined
        structural-frequency changes, and suppress noise response when the teacher supports doing so.
        For $D_{h,k}=.01[a_h(x_k)-a_h(x)]/\max(\mathrm{RMS}(x_k-x),.001)$:
        $$L_{SSD}=E[w\,\mathrm{Huber}(D_{S,str}-D_{T,str})+
        \gamma w e^{-|D_{T,noi}|}\mathrm{Huber}(D_{S,noi})].$$
        Train with $CE+\alpha\tau^2 KL(T\Vert S)+\beta L_{SSD}$.
        See `docs/METHOD.md` for the intervention definition, gates, derivation and prior art.
        This is a testable research proposal; novelty and improvement are not established.
        """)] + common() + training_helpers()
    cells += [code('''
        STUDENT_ARCH = "mobilenet_v3_small"  # alternative: efficientnet_b0
        STUDENT_EPOCHS = 30
        PARSER = RUN_ROOT / "parser" / "best.pt"
        TEACHER = RUN_ROOT / "teacher_freqblender" / "best.pt"
        assert PARSER.exists() and TEACHER.exists(), "Complete notebook 01 first."
        common_args = ["--parser", PARSER, "--teacher", TEACHER, "--temperature", 4,
                       "--kd-weight", .5, "--strength", .25, "--epsilon", .02]
        '''), md("""
        ## Controlled ablations
        All students share initialization seed, crop resolution, training schedule, and
        FreqBlender augmentation. CE and ordinary KD isolate architectural and standard
        distillation effects. Structural-only isolates the additional noise term.
        Training all four takes longer; run each in separate Kaggle sessions if needed.
        Do not choose a variant based on test scores.
        """), code('''
        train_stage("student", "student_ce", STUDENT_EPOCHS, STUDENT_ARCH,
                    extra=common_args + ["--mode", "ce"])
        '''), code('''
        train_stage("student", "student_kd", STUDENT_EPOCHS, STUDENT_ARCH,
                    extra=common_args + ["--mode", "kd"])
        '''), code('''
        train_stage("student", "student_structure", STUDENT_EPOCHS, STUDENT_ARCH,
                    extra=common_args + ["--mode", "proposed", "--noise-weight", 0])
        '''), code('''
        train_stage("student", "student_proposed", STUDENT_EPOCHS, STUDENT_ARCH,
                    extra=common_args + ["--mode", "proposed", "--noise-weight", .25])
        '''), code('''
        import pandas as pd
        import matplotlib.pyplot as plt
        rows = []
        for name in ("student_ce", "student_kd", "student_structure", "student_proposed"):
            history = pd.read_json(RUN_ROOT/name/"history.json")
            plt.plot(history.epoch, history.validation_score, label=name)
            rows.append({"run": name, "best_validation_image_auc": history.validation_score.max()})
        plt.xlabel("Epoch"); plt.ylabel("Validation image AUC"); plt.legend(); plt.grid(alpha=.2)
        display(pd.DataFrame(rows))
        '''), md("""
        Repeat with seeds 43 and 44 in separate run roots and matched teachers to assess variability.
        Tune the secant weight/strength only on validation. Report unsuccessful variants too.
        The proposed student has exactly the same inference architecture as the CE/KD students.
        Open **03_evaluate_and_export.ipynb** once training and model choices are fixed.
        """)]
    cells.insert(1, md("This PNG dataset has no supplied video/source grouping. All video-named "
                       "metrics, validation scores, thresholds and bootstrap groups below operate "
                       "on individual images. They must be reported as image-level results, "
                       "not source-video results. Existing folder splits are preserved."))
    return cells


def evaluation():
    cells = [md("""
        # 03 · Final evaluation, inference timing and export
        Evaluate fixed checkpoints on held-out images. Thresholds are selected on validation
        images only. Bootstrap intervals resample images, since source-video groups are unavailable. No invented results
        are included: every table below is produced from your checkpoints and data.
        """)] + common()
    cells += [code('''
        from freqblender.data import read_manifest
        table = read_manifest(MANIFEST)
        RUN_NAMES = ["teacher_spatial", "teacher_freqblender", "student_ce", "student_kd",
                     "student_structure", "student_proposed"]
        AVAILABLE = [name for name in RUN_NAMES if (RUN_ROOT/name/"best.pt").exists()]
        assert "teacher_freqblender" in AVAILABLE and "student_proposed" in AVAILABLE
        print("Comparing:", AVAILABLE)
        # Same hardware, input size and precision are required for a matched comparison.
        configs = {name: torch.load(RUN_ROOT/name/"best.pt", map_location="cpu", weights_only=True)["config"]
                   for name in AVAILABLE}
        assert len({cfg["size"] for cfg in configs.values()}) == 1
        assert len({cfg["manifest_sha256"] for cfg in configs.values()}) == 1
        '''), md("""
        ## Evaluate each model in a separate process
        Each process holds only one detector, making peak GPU-memory comparisons meaningful.
        Model-only timing includes normalization and forward execution, with CUDA warmup and
        synchronization. It excludes face detection, decoding, resizing and transfer. All runs
        use FP32 at batch sizes 1 and 16; change 16 consistently if GPU memory requires it.
        """), code('''
        for name in AVAILABLE:
            subprocess.run([sys.executable, "-m", "freqblender.evaluate",
                            "--checkpoint", str(RUN_ROOT/name/"best.pt"),
                            "--manifest", str(MANIFEST), "--output", str(RUN_ROOT/name/"evaluation"),
                            "--batch-size", "16", "--bootstrap", "1000"], check=True)
        '''), code('''
        import pandas as pd
        reports, rows = {}, []
        for name in AVAILABLE:
            report = json.loads((RUN_ROOT/name/"evaluation/metrics.json").read_text())
            reports[name] = report
            single = report["benchmark"][0]
            throughput = report["benchmark"][1]
            for dataset, result in report["datasets"].items():
                rows.append(dict(run=name, dataset=dataset, image_auc=result["video"]["auc"],
                                 frame_auc=result["frame"]["auc"], auc_ci_low=result["video_auc_ci"]["low"],
                                 auc_ci_high=result["video_auc_ci"]["high"],
                                 balanced_accuracy=result["video"]["balanced_accuracy"],
                                 parameters=single["parameters"], batch1_ms=single["median_batch_ms"],
                                 batch1_p95_ms=single["p95_batch_ms"], throughput=throughput["images_per_second"],
                                 peak_cuda_mb=throughput["peak_cuda_mb"]))
        results = pd.DataFrame(rows)
        teacher_ms = reports["teacher_freqblender"]["benchmark"][0]["median_batch_ms"]
        results["speedup_vs_teacher"] = teacher_ms/results.batch1_ms
        results.to_csv(RUN_ROOT/"comparison.csv", index=False)
        display(results)
        '''), md("""
        ## Paired AUC differences against ordinary KD
        The proposed loss should be judged against the same compact architecture with
        ordinary KD, not only against the larger teacher. These intervals reuse each
        sampled image for both models. An interval crossing zero does not establish
        an AUC gain. This accounts for test-set sampling, not training-seed variation.
        """), code('''
        from freqblender.evaluate import paired_auc_difference
        if "student_kd" in AVAILABLE:
            proposed = pd.read_csv(RUN_ROOT/"student_proposed/evaluation/test_video_predictions.csv")
            reference = pd.read_csv(RUN_ROOT/"student_kd/evaluation/test_video_predictions.csv")
            differences = {}
            for dataset in proposed.dataset.unique():
                differences[dataset] = paired_auc_difference(
                    proposed[proposed.dataset == dataset], reference[reference.dataset == dataset])
            display(pd.DataFrame(differences).T)
            (RUN_ROOT/"paired_auc_difference_vs_kd.json").write_text(json.dumps(differences, indent=2))
        '''), code('''
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(9, 5))
        for _, row in results[results.dataset == "FF++"].iterrows():
            axis.scatter(row.batch1_ms, row.image_auc, s=70)
            axis.annotate(row.run, (row.batch1_ms, row.image_auc), xytext=(5, 5), textcoords="offset points", fontsize=8)
        axis.set_xlabel("Median batch-1 model latency (ms, lower is better)")
        axis.set_ylabel("Held-out image AUC (higher is better)")
        axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(RUN_ROOT/"accuracy_latency.png", dpi=180)
        '''), md("""
        ## Export the compact detector
        Export contains only the student and input normalization. No DCT, FPNet, auxiliary
        network or teacher is needed. The sidecar describes input/output conventions.
        Detection thresholds remain in each run's evaluation report.
        """), code('''
        from freqblender.evaluate import export_detector
        EXPORT = RUN_ROOT / "student_proposed" / "student.ts"
        export_detector(RUN_ROOT/"student_proposed/best.pt", EXPORT)
        print("Export:", EXPORT, "Size MB:", EXPORT.stat().st_size/2**20)
        '''), code('''
        import numpy as np
        from PIL import Image
        CROP_PATH = Path(table[table.split == "test"].iloc[0].path)  # replace with YOUR FACE CROP
        size = configs["student_proposed"]["size"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        deployed = torch.jit.load(str(EXPORT), map_location=device).eval()
        with Image.open(CROP_PATH) as image:
            image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.array(image, dtype=np.float32)/255).permute(2,0,1)[None].to(device)
        with torch.inference_mode():
            fake_score = deployed(tensor).softmax(1)[0,1].item()
        threshold = reports["student_proposed"]["datasets"]["FF++"]["video"]["threshold"]
        print({"fake_score": fake_score, "validation_image_threshold": threshold,
               "thresholded_prediction": "fake" if fake_score >= threshold else "real"})
        # This threshold is calibrated on validation images, not videos.
        '''), md("""
        For external-dataset testing, append its face-crop rows as `test` with a unique dataset
        name; retain FF++ validation. Inspect `metrics.json` for AP, low-FPR TPR and timing
        metadata. Compare multiple seeds and paired group-bootstrap differences before claiming
        small gains. FF++ results alone do not establish cross-dataset generalization.
        """)]
    cells.insert(1, md("This PNG dataset has no supplied video/source grouping. All video-named "
                       "metrics, validation scores, thresholds and bootstrap groups below operate "
                       "on individual images. They must be reported as image-level results, "
                       "not source-video results. Existing folder splits are preserved."))
    return cells


def main():
    destination = ROOT / "notebooks"
    destination.mkdir(exist_ok=True)
    builders = [("00_prepare_ffpp", prepare), ("01_train_freqblender", baseline),
                ("02_train_efficient_student", student), ("03_evaluate_and_export", evaluation)]
    for name, builder in builders:
        notebook = {"nbformat": 4, "nbformat_minor": 5,
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python", "version": "3.12"},
                                 "colab": {"name": name+".ipynb", "provenance": []}, "accelerator": "GPU"},
                    "cells": builder()}
        for i, cell in enumerate(notebook["cells"]):
            cell["id"] = f"cell-{i:03d}"
        (destination / (name+".ipynb")).write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
        print(name, len(notebook["cells"]), "cells")


if __name__ == "__main__":
    main()
