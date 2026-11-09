from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


REQUIRED_COLUMNS = {"path", "label", "split", "video_id", "group_id", "dataset"}


def build_image_manifest(root, destination):
    """Index existing split/{real,fake}/*.png without guessing video identities."""
    root, destination = Path(root).resolve(), Path(destination)
    records = []
    for split in ("train", "val", "test"):
        for category, label in (("real", 0), ("fake", 1)):
            folder = root / split / category
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected image directory: {folder}")
            paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png")
            if not paths:
                raise ValueError(f"No PNG images in {folder}")
            for path in paths:
                identifier = path.relative_to(root).as_posix()
                records.append(dict(path=str(path), label=label, split=split,
                                    video_id=identifier, group_id=identifier,
                                    dataset="FF++", sample_unit="image"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(destination, index=False)
    return destination


def read_manifest(path, root=None, check_files=True):
    table = pd.read_csv(path, dtype={"video_id": str, "group_id": str, "dataset": str})
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if table.empty or table[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Manifest is empty or contains missing required values.")
    if not table.label.isin([0, 1]).all():
        raise ValueError("Labels must be 0=real, 1=fake.")
    if not table.split.isin(["train", "val", "test"]).all():
        raise ValueError("Use train/val/test splits.")
    # Dataset names namespace videos/groups across external test sets.
    for key in ("group_id", "video_id"):
        if (table.groupby(["dataset", key]).split.nunique() > 1).any():
            raise ValueError(f"Data leakage: {key} occurs in multiple splits.")
    if (table.groupby(["dataset", "video_id"]).label.nunique() > 1).any():
        raise ValueError("A video has inconsistent labels.")
    base = Path(root) if root else Path(path).resolve().parent
    table["path"] = [str((base / str(p)).resolve()) for p in table.path]
    if table.path.duplicated().any():
        raise ValueError("Duplicate frame paths in manifest.")
    if check_files:
        absent = [p for p in table.path if not Path(p).is_file()]
        if absent:
            raise FileNotFoundError(f"{len(absent)} missing crops, e.g. {absent[:3]}")
    return table


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FaceFrames(Dataset):
    def __init__(self, table, size=224, augment=False):
        self.table = table.reset_index(drop=True)
        self.size, self.augment = size, augment
        if self.table.empty:
            raise ValueError("Selected dataset split is empty.")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        with Image.open(row.path) as image:
            image = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
            array = np.array(image, dtype=np.float32) / 255.
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if self.augment and torch.rand(()) < .5:
            tensor = tensor.flip(-1)
        return {"image": tensor, "label": int(row.label),
                "video_id": row.video_id, "dataset": row.dataset, "group_id": row.group_id}


def make_loader(table, size, batch_size, train=False, workers=2, seed=42):
    if train and len(table) < 2:
        raise ValueError("Training requires at least two frames for batch normalization.")
    batch_size = min(batch_size, len(table))
    return DataLoader(FaceFrames(table, size, train), batch_size=batch_size,
                      shuffle=train, num_workers=workers, drop_last=train,
                      pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed))
