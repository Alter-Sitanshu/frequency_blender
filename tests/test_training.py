"""Actual small CPU optimizer/checkpoint runs; no dataset/download required."""
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from freqblender.train import argument_parser, resolve_device_and_ids, run_training
from freqblender.frequency import FPNet
from freqblender.models import Detector


def make_fixture(tmp_path):
    records = []
    rng = np.random.default_rng(1)
    for split in ["train", "val", "test"]:
        for label in [0, 1]:
            for i in range(2):
                path = tmp_path / f"{split}_{label}_{i}.png"
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
                records.append(dict(path=path.name, label=label, split=split, video_id=f"{split}_{label}",
                                    group_id=split, dataset="fixture"))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    return manifest


def cfg(manifest, output, stage="teacher", architecture="mobilenet_v3_small", extra=()):
    return vars(argument_parser().parse_args([
        "--manifest", str(manifest), "--output", str(output), "--stage", stage,
        "--architecture", architecture, "--size", "32", "--width", "2", "--batch-size", "2",
        "--epochs", "1", "--workers", "0", "--max-steps", "1", "--no-pretrained", *map(str, extra)]))


def test_teacher_resume_and_all_student_objectives(tmp_path):
    manifest = make_fixture(tmp_path)
    teacher = run_training(cfg(manifest, tmp_path/"teacher"))
    configuration = cfg(manifest, tmp_path/"teacher", extra=["--resume"])
    configuration["epochs"] = 2
    run_training(configuration)
    history = json.loads((tmp_path/"teacher/history.json").read_text())
    assert [item["epoch"] for item in history] == [1, 2]
    uninterrupted_cfg = cfg(manifest, tmp_path/"uninterrupted")
    uninterrupted_cfg["epochs"] = 2
    run_training(uninterrupted_cfg)
    resumed = torch.load(tmp_path/"teacher/last.pt", weights_only=True)
    uninterrupted = torch.load(tmp_path/"uninterrupted/last.pt", weights_only=True)
    for key, value in resumed["model"].items():
        torch.testing.assert_close(value, uninterrupted["model"][key], rtol=0, atol=0)
    parser = FPNet(32, 2)
    parser_path = tmp_path/"parser.pt"
    torch.save({"model": parser.state_dict(), "config": {"stage": "parser", "size": 32, "width": 2}}, parser_path)
    for mode in ["ce", "kd", "proposed"]:
        path = run_training(cfg(manifest, tmp_path/mode, "student", extra=[
            "--mode", mode, "--teacher", teacher, "--parser", parser_path]))
        checkpoint = torch.load(path, weights_only=True)
        assert np.isfinite(checkpoint["history"][0]["train_loss"])
        assert checkpoint["config"]["mode"] == mode


def test_parser_optimizer_with_external_identity(tmp_path):
    manifest = make_fixture(tmp_path)
    # Actual ResNet34 auxiliary; random weights are a test fixture, never a trained claim.
    discriminator = Detector("resnet34", pretrained=False)
    discriminator_path = tmp_path/"disc.pt"
    torch.save({"model": discriminator.state_dict(), "config": {"stage": "warmup", "architecture": "resnet34", "size": 32}}, discriminator_path)
    identity = nn.Sequential(nn.Conv2d(3, 8, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    identity_path = tmp_path/"identity.ts"
    torch.jit.trace(identity, torch.rand(2, 3, 32, 32)).save(str(identity_path))
    path = run_training(cfg(manifest, tmp_path/"parser", "parser", extra=[
        "--discriminator", discriminator_path, "--identity-checkpoint", identity_path, "--identity-size", "32"]))
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["config"]["identity_kind"] == "external_arcface"
    assert np.isfinite(checkpoint["best_score"])


def test_gpu_selection_defaults_to_two_and_caps_at_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    device, device_ids = resolve_device_and_ids(2)
    assert device == torch.device("cuda:0")
    assert device_ids == [0, 1]

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_device_and_ids(2)[1] == [0]


def test_facenet_adapter_architecture_and_input_gradients(monkeypatch):
    import pytest
    facenet = pytest.importorskip("facenet_pytorch")
    from freqblender.models import IdentityEncoder
    factory = facenet.InceptionResnetV1
    # Avoid pretrained downloads in tests; exercise the actual architecture/wrapper.
    monkeypatch.setattr(facenet, "InceptionResnetV1", lambda pretrained: factory(pretrained=None))
    identity = IdentityEncoder().eval()
    x = torch.rand(2, 3, 160, 160, requires_grad=True)
    embeddings = identity(x)
    assert embeddings.shape == (2, 512)
    embeddings[:, 0].sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in identity.parameters())
