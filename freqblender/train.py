"""Four stages: warmup ResNet34, FPNet, detector teacher, compact student."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .augment import training_pair, spatial_self_blend
from .data import read_manifest, manifest_hash, make_loader
from .frequency import FPNet
from .models import Detector, IdentityEncoder, freeze
from .losses import parser_loss, distillation_kl, counterfactual_views, secant_loss
from .evaluate import load_detector, predict, video_predictions, metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_parser(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    if cfg["stage"] != "parser":
        raise ValueError("Expected FPNet checkpoint.")
    parser = FPNet(cfg["size"], cfg["width"])
    parser.load_state_dict(checkpoint["model"])
    return freeze(parser.to(device))


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@torch.no_grad()
def validate_parser(model, loader, identity, discriminator, device):
    model.eval()
    losses, count = 0., 0
    # Hold random self-blends fixed across epochs, preserve training RNG state.
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(1701)
        for batch in loader:
            real = batch["image"].to(device)
            loss, _ = parser_loss(model, real, spatial_self_blend(real), identity, discriminator)
            losses += float(loss) * len(real)
            count += len(real)
    return -losses/count  # higher is better, as for AUC


def run_training(config):
    cfg = dict(config)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    table = read_manifest(cfg["manifest"])
    cfg["manifest_sha256"] = manifest_hash(cfg["manifest"])
    # Real-only synthetic training. Actual FF++ fakes are used only for validation/test.
    train_rows = table[(table.split == "train") & (table.label == 0)]
    val_rows = table[table.split == "val"]
    if cfg["stage"] == "parser":
        val_rows = val_rows[val_rows.label == 0]
    if cfg["batch_size"] < 2:
        raise ValueError("Training batch_size must be >=2.")
    train_loader = make_loader(train_rows, cfg["size"], cfg["batch_size"], True, cfg["workers"], cfg["seed"])
    val_loader = make_loader(val_rows, cfg["size"], cfg["batch_size"], False, cfg["workers"])
    parser = identity = discriminator = teacher = None
    if cfg["stage"] == "parser":
        if not cfg["discriminator"]:
            raise ValueError("Train warmup discriminator first, then pass --discriminator.")
        model = FPNet(cfg["size"], cfg["width"]).to(device)
        discriminator, disc_ckpt = load_detector(cfg["discriminator"], device)
        if disc_ckpt["config"]["architecture"] != "resnet34":
            raise ValueError("FPNet authenticity auxiliary must be ResNet34.")
        freeze(discriminator)
        identity = freeze(IdentityEncoder(cfg["identity_checkpoint"], cfg["identity_size"]).to(device))
        cfg["identity_kind"] = identity.kind
    else:
        if cfg["stage"] == "warmup" and cfg["architecture"] != "resnet34":
            raise ValueError("Use --architecture resnet34 for warmup.")
        model = Detector(cfg["architecture"], cfg["pretrained"]).to(device)
        if cfg["parser"]:
            parser = load_parser(cfg["parser"], device)
            if parser.size != cfg["size"]:
                raise ValueError("Parser and detector training image sizes must match.")
        if cfg["stage"] == "student" and cfg["mode"] != "ce":
            if not cfg["teacher"]:
                raise ValueError("KD requires --teacher checkpoint.")
            teacher, teacher_ckpt = load_detector(cfg["teacher"], device)
            freeze(teacher)
            if teacher_ckpt["config"]["size"] != cfg["size"]:
                raise ValueError("Use the teacher's image size for a controlled comparison.")
            if cfg["mode"] == "proposed" and parser is None:
                raise ValueError("Proposed distillation requires --parser.")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # FPNet/identity gradients use FP32. AMP is optional for detector/student stages.
    amp = cfg["amp"] and device.type == "cuda" and cfg["stage"] != "parser"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_score, history = 0, -float("inf"), []
    resume = out / "last.pt"
    if cfg["resume"] and resume.exists():
        old = torch.load(resume, map_location="cpu", weights_only=True)
        ignored = {"epochs", "workers", "resume", "output", "manifest"}
        for key, value in old["config"].items():
            if key not in ignored and cfg.get(key) != value:
                raise ValueError(f"Resume configuration mismatch for {key}; use a new output directory.")
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        scaler.load_state_dict(old["scaler"])
        start_epoch, best_score, history = old["epoch"]+1, old["best_score"], old["history"]
        restore_rng(old["rng"])
    elif resume.exists():
        raise FileExistsError("Output has checkpoints; use --resume or a new output directory.")
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for epoch in range(start_epoch, cfg["epochs"]):
        train_loader.generator.manual_seed(cfg["seed"] + epoch)
        model.train()
        total, samples = 0., 0
        progress = tqdm(train_loader, desc=f"{cfg['stage']} {epoch+1}/{cfg['epochs']}")
        for step, batch in enumerate(progress):
            if cfg["max_steps"] and step >= cfg["max_steps"]:
                break
            real = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if cfg["stage"] == "parser":
                loss, _ = parser_loss(model, real, spatial_self_blend(real), identity, discriminator)
            else:
                fake = training_pair(real, parser, cfg["blend_probability"])
                x = torch.cat([real, fake])
                y = torch.cat([torch.zeros(len(real)), torch.ones(len(real))]).long().to(device)
                with torch.autocast(device.type, enabled=amp):
                    if cfg["stage"] == "student" and cfg["mode"] == "proposed":
                        # BatchNorm sees base and interventions in the SAME forward pass.
                        # Opposite-label synthetic donors share the original identity.
                        donor = torch.cat([fake, real])
                        xs, xn = counterfactual_views(x, donor, parser,
                                                       cfg["strength"], cfg["epsilon"])
                        all_x = torch.cat([x, xs, xn])
                        s0, ss, sn = model(all_x).chunk(3)
                        with torch.no_grad():
                            t0, ts, tn = teacher(all_x).chunk(3)
                        loss = F.cross_entropy(s0, y) + cfg["kd_weight"]*distillation_kl(s0, t0, cfg["temperature"])
                        # Compute small logit differences in FP32.
                        loss += cfg["secant_weight"]*secant_loss(
                            s0.float(), ss.float(), sn.float(), t0.float(), ts.float(), tn.float(), y,
                            x, xs, xn, cfg["noise_weight"])
                    else:
                        logits = model(x)
                        loss = F.cross_entropy(logits, y)
                        if teacher is not None:
                            with torch.no_grad():
                                target = teacher(x)
                            loss += cfg["kd_weight"]*distillation_kl(logits, target, cfg["temperature"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss. Inspect data and auxiliary checkpoints.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())*len(real)
            samples += len(real)
            progress.set_postfix(loss=f"{total/samples:.4f}")
        if cfg["stage"] == "parser":
            score = validate_parser(model, val_loader, identity, discriminator, device)
        else:
            predictions = predict(model, val_loader, device)
            score = metrics(video_predictions(predictions))["auc"]
        improved = score > best_score
        best_score = max(score, best_score)
        history.append(dict(epoch=epoch+1, train_loss=total/samples, validation_score=score))
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                       epoch=epoch, best_score=best_score, history=history, config=cfg, rng=rng_state())
        atomic_save(payload, out / "last.pt")
        if improved:
            atomic_save(payload, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"validation score={score:.5f}; best={best_score:.5f}")
    return out / "best.pt"


def argument_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["warmup", "parser", "teacher", "student"], required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--architecture", default="efficientnet_b4")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=0, help="Smoke tests only; 0 uses all batches.")
    p.add_argument("--parser", default=None)
    p.add_argument("--discriminator", default=None)
    p.add_argument("--identity-checkpoint", default=None)
    p.add_argument("--identity-size", type=int, default=112)
    p.add_argument("--teacher", default=None)
    p.add_argument("--mode", choices=["ce", "kd", "proposed"], default="proposed")
    p.add_argument("--blend-probability", type=float, default=.2)
    p.add_argument("--temperature", type=float, default=4.)
    p.add_argument("--kd-weight", type=float, default=.5)
    p.add_argument("--secant-weight", type=float, default=1.)
    p.add_argument("--noise-weight", type=float, default=.25)
    p.add_argument("--strength", type=float, default=.25)
    p.add_argument("--epsilon", type=float, default=.02)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.set_defaults(pretrained=True)
    return p


if __name__ == "__main__":
    print(run_training(vars(argument_parser().parse_args())))
