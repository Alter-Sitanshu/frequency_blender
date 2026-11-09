import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve
from .models import Detector
from .data import read_manifest, make_loader


def load_detector(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config"]["stage"] == "parser":
        raise ValueError("Expected a detector checkpoint, not FPNet.")
    model = Detector(checkpoint["config"]["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    records = []
    for batch in loader:
        probability = model(batch["image"].to(device)).softmax(1)[:, 1].cpu().numpy()
        for i, score in enumerate(probability):
            records.append(dict(score=float(score), label=int(batch["label"][i]),
                                video_id=batch["video_id"][i], group_id=batch["group_id"][i],
                                dataset=batch["dataset"][i]))
    return pd.DataFrame(records)


def video_predictions(frame_predictions):
    return frame_predictions.groupby(["dataset", "video_id", "group_id"], as_index=False).agg(
        score=("score", "mean"), label=("label", "first"), frames=("score", "size"))


def metrics(predictions, threshold=.5):
    y, p = predictions.label.to_numpy(), predictions.score.to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("Evaluation requires both real and fake samples.")
    fpr, tpr, _ = roc_curve(y, p)
    return dict(auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, p >= threshold)),
                tpr_at_fpr_1pct=float(tpr[fpr <= .01].max()), threshold=float(threshold), n=len(y))


def select_threshold(validation_videos):
    if validation_videos.label.nunique() != 2:
        raise ValueError("Threshold calibration needs both classes in validation.")
    fpr, tpr, thresholds = roc_curve(validation_videos.label, validation_videos.score)
    valid = np.isfinite(thresholds)
    return float(thresholds[valid][np.argmax((tpr-fpr)[valid])])


def group_bootstrap_auc(videos, repetitions=1000, seed=42):
    """Resample original/manipulated PAIRS, retaining within-group dependence."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in videos.groupby(["dataset", "group_id"])]
    values = []
    for _ in range(repetitions):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if sample.label.nunique() == 2:
            values.append(roc_auc_score(sample.label, sample.score))
    if not values:
        return {"low": None, "high": None, "valid_resamples": 0}
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"low": float(lower), "high": float(upper), "valid_resamples": len(values)}


def paired_auc_difference(candidate, reference, repetitions=1000, seed=42):
    """Paired group bootstrap of AUC(candidate) - AUC(reference)."""
    keys = ["dataset", "video_id", "group_id"]
    if candidate.duplicated(keys).any() or reference.duplicated(keys).any():
        raise ValueError("Expected one prediction per video.")
    merged = candidate.merge(reference, on=keys, suffixes=("_candidate", "_reference"),
                             how="outer", indicator=True, validate="one_to_one")
    if not (merged["_merge"] == "both").all() or not (merged.label_candidate == merged.label_reference).all():
        raise ValueError("Paired comparisons require identical videos and labels.")
    if merged.dataset.nunique() != 1 or merged.label_candidate.nunique() != 2:
        raise ValueError("Compare one dataset at a time with both classes.")
    groups = [group for _, group in merged.groupby("group_id")]
    rng = np.random.default_rng(seed)
    def difference(frame):
        labels = frame.label_candidate
        return roc_auc_score(labels, frame.score_candidate) - roc_auc_score(labels, frame.score_reference)
    values = []
    for _ in range(repetitions):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if sample.label_candidate.nunique() == 2:
            values.append(difference(sample))
    low, high = np.percentile(values, [2.5, 97.5]) if values else (None, None)
    return {"auc_difference": float(difference(merged)),
            "low": float(low) if low is not None else None,
            "high": float(high) if high is not None else None,
            "valid_resamples": len(values)}


@torch.inference_mode()
def benchmark(model, size, device, batch_size=1, warmup=20, repeats=100, amp=False):
    """Model-only wall latency with synchronization; excludes data transfer/cropping."""
    device = torch.device(device)
    model.eval().to(device)
    x = torch.rand(batch_size, 3, size, size, device=device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def run():
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            return model(x)
    for _ in range(warmup):
        run()
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        run()
        sync()
        durations.append((time.perf_counter()-start)*1000)
    median = float(np.median(durations))
    return dict(batch_size=batch_size, image_size=size, median_batch_ms=median,
                p95_batch_ms=float(np.percentile(durations, 95)),
                median_ms_per_image=median/batch_size, images_per_second=1000*batch_size/median,
                parameters=sum(p.numel() for p in model.parameters()),
                peak_cuda_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None,
                precision="autocast_fp16" if amp and device.type == "cuda" else "fp32",
                device=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                torch=str(torch.__version__), warmup=warmup, repeats=repeats,
                timing_scope="model only; includes normalization, excludes loading/crops/transfers")


def export_detector(checkpoint_path, destination, size=None):
    model, checkpoint = load_detector(checkpoint_path)
    size = size or checkpoint["config"]["size"]
    example = torch.rand(1, 3, size, size)
    with torch.inference_mode():
        exported = torch.jit.trace(model, example)
        exported = torch.jit.freeze(exported)
        torch.testing.assert_close(exported(example), model(example), atol=1e-5, rtol=1e-4)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported.save(str(destination))
    destination.with_suffix(".json").write_text(json.dumps({
        "input": f"float32 RGB [N,3,{size},{size}], values [0,1], bilinear resize",
        "output": "two logits, index 0 real / index 1 fake; softmax to probabilities",
        "normalization": "included in exported model", "config": checkpoint["config"]}, indent=2))
    return destination


def evaluate_checkpoint(checkpoint_path, manifest, output, batch_size=16, workers=2, bootstrap=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_detector(checkpoint_path, device)
    size = checkpoint["config"]["size"]
    table = read_manifest(manifest)
    val = predict(model, make_loader(table[table.split == "val"], size, batch_size, workers=workers), device)
    threshold = select_threshold(video_predictions(val))
    test = predict(model, make_loader(table[table.split == "test"], size, batch_size, workers=workers), device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    val.to_csv(output / "validation_frame_predictions.csv", index=False)
    test.to_csv(output / "test_frame_predictions.csv", index=False)
    videos = video_predictions(test)
    videos.to_csv(output / "test_video_predictions.csv", index=False)
    image_only = "sample_unit" in table and table.sample_unit.eq("image").all()
    report = {"checkpoint": str(checkpoint_path),
              "evaluation_unit": "image" if image_only else "video",
              "grouping_note": "Image IDs only; video/source grouping unavailable" if image_only else "Manifest video/source groups",
              "threshold_source": "validation image Youden J" if image_only else "validation video Youden J",
              "datasets": {}, "config": checkpoint["config"]}
    for dataset, subset in videos.groupby("dataset"):
        report["datasets"][dataset] = {
            "video": metrics(subset, threshold), "video_auc_ci": group_bootstrap_auc(subset, bootstrap),
            "frame": metrics(test[test.dataset == dataset], threshold)}
    # Measure one resident detector at a time; consistent size/precision in comparisons.
    report["benchmark"] = [benchmark(model, size, device, batch_size=b) for b in (1, batch_size)]
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.manifest, args.output,
                                         args.batch_size, args.workers, args.bootstrap), indent=2))


if __name__ == "__main__":
    main()
