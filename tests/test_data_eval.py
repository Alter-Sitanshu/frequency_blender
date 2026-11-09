import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from freqblender.data import read_manifest, make_loader, build_image_manifest
from freqblender.preprocess import official_splits
from freqblender.evaluate import video_predictions, metrics, select_threshold, group_bootstrap_auc, paired_auc_difference, benchmark, export_detector
from freqblender.models import Detector


def test_presplit_png_manifest_preserves_splits_and_labels(tmp_path):
    root = tmp_path / "input"
    for split in ("train", "val", "test"):
        for category in ("real", "fake"):
            folder = root / split / category
            folder.mkdir(parents=True)
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(folder / "same_name.png")
    destination = tmp_path / "output" / "manifest.csv"
    build_image_manifest(root, destination)
    table = read_manifest(destination)
    assert len(table) == 6 and table.video_id.nunique() == 6
    assert set(table.sample_unit) == {"image"}
    for row in table.itertuples():
        assert Path(row.path).parent.parent.name == row.split
        assert row.label == int(Path(row.path).parent.name == "fake")
    before = destination.read_bytes()
    build_image_manifest(root, destination)
    assert destination.read_bytes() == before
    with pytest.raises(FileNotFoundError, match="Expected image directory"):
        build_image_manifest(tmp_path / "absent", destination)


def test_manifest_leakage_checks(tmp_path):
    image = tmp_path / "face.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image)
    rows = [dict(path="face.png", label=0, split="train", video_id="one", group_id="pair", dataset="FF++"),
            dict(path="other.png", label=1, split="test", video_id="two", group_id="pair", dataset="FF++")]
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="leakage"):
        read_manifest(manifest, check_files=False)
    rows[1]["group_id"] = "other"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    with pytest.raises(FileNotFoundError):
        read_manifest(manifest)
    table = read_manifest(manifest, check_files=False)
    assert table.path.iloc[0] == str(image.resolve())


def test_official_pair_grouping(tmp_path):
    for split, pair in zip(["train", "val", "test"], [["000", "003"], ["001", "004"], ["002", "005"]]):
        (tmp_path / f"{split}.json").write_text(json.dumps([pair]))
    mapping = official_splits(tmp_path)
    assert mapping["000"] == mapping["003"] == ("train", "000_003")
    (tmp_path / "test.json").write_text(json.dumps([["000", "005"]]))
    with pytest.raises(ValueError, match="Conflicting"):
        official_splits(tmp_path)


def test_ffpp_crop_pipeline_with_shared_original_boxes(tmp_path, monkeypatch):
    import cv2
    from freqblender.preprocess import extract_ffpp
    splits = tmp_path / "splits"
    splits.mkdir()
    root, output = tmp_path/"ffpp", tmp_path/"prepared"
    class FixedDetector:
        def detectMultiScale(self, *args, **kwargs):
            return np.array([[20, 15, 60, 65]])
    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *args: FixedDetector())
    for split, pair in zip(["train", "val", "test"], [["000", "003"], ["001", "004"], ["002", "005"]]):
        (splits/f"{split}.json").write_text(json.dumps([pair]))
        paths = [root/f"original_sequences/youtube/c23/videos/{member}.mp4" for member in pair]
        paths.append(root/f"manipulated_sequences/Deepfakes/c23/videos/{pair[0]}_{pair[1]}.mp4")
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 96))
            if not writer.isOpened():
                pytest.skip("System OpenCV lacks MP4 encoder")
            for i in range(3):
                writer.write(np.full((96, 96, 3), 80+i*20, dtype=np.uint8))
            writer.release()
    manifest = extract_ffpp(root, splits, output, frames=2, crop_size=32, methods=["Deepfakes"])
    table = read_manifest(manifest)
    assert len(table) == 18 and set(table.split) == {"train", "val", "test"}
    assert set(table.label) == {0, 1}
    assert Image.open(table.path.iloc[0]).size == (32, 32)
    fake = table[table.video_id == "Deepfakes/000_003"]
    real = table[table.video_id == "original/000"]
    assert fake.group_id.iloc[0] == real.group_id.iloc[0] == "000_003"
    assert list(fake.frame_index) == list(real.frame_index)


def test_metrics_aggregate_videos_not_frames():
    frames = pd.DataFrame(dict(score=[.1, .2, .8, .9, .4, .6], label=[0, 0, 1, 1, 0, 1],
                               video_id=["a", "a", "b", "b", "c", "d"],
                               group_id=["g1", "g1", "g1", "g1", "g2", "g2"], dataset=["FF++"]*6))
    videos = video_predictions(frames)
    assert len(videos) == 4
    assert metrics(videos)["auc"] == 1.
    assert .4 < select_threshold(videos) <= .6
    assert group_bootstrap_auc(videos, repetitions=20) == {"low": 1., "high": 1., "valid_resamples": 20}
    paired = paired_auc_difference(videos, videos, repetitions=20)
    assert paired["auc_difference"] == paired["low"] == paired["high"] == 0
    with pytest.raises(ValueError, match="identical"):
        paired_auc_difference(videos, videos.iloc[:-1], repetitions=2)
    with pytest.raises(ValueError, match="both"):
        metrics(videos[videos.label == 0])


def test_real_mobilenet_export_and_batch_inference(tmp_path):
    model = Detector("mobilenet_v3_small", pretrained=False).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "config": {"stage": "student", "architecture": "mobilenet_v3_small", "size": 32}}, checkpoint)
    exported = export_detector(checkpoint, tmp_path / "student.ts")
    restored = torch.jit.load(str(exported)).eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), atol=1e-5, rtol=1e-4)
    report = benchmark(model, 32, "cpu", warmup=1, repeats=3)
    assert report["median_batch_ms"] > 0 and report["parameters"] > 0


@pytest.mark.parametrize("architecture", ["resnet34", "efficientnet_b4", "efficientnet_b0"])
def test_real_detector_forward(architecture):
    model = Detector(architecture, pretrained=False).eval()
    with torch.no_grad():
        y = model(torch.rand(2, 3, 32, 32))
    assert y.shape == (2, 2) and torch.isfinite(y).all()
