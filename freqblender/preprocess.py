"""FF++ official video splits -> common face crops and auditable CSV manifest."""
import argparse
import json
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm


def official_splits(split_dir):
    assignment = {}
    for split in ("train", "val", "test"):
        path = Path(split_dir) / f"{split}.json"
        pairs = json.loads(path.read_text())
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Invalid FF++ pair in {path}: {pair}")
            # Official pairs define groups; use the same group for both originals
            # and both directions of their manipulated videos.
            members = [str(item).zfill(3) for item in pair]
            group = "_".join(sorted(members))
            for video in members:
                value = (split, group)
                if video in assignment and assignment[video] != value:
                    raise ValueError(f"Conflicting official split/group for {video}")
                assignment[video] = value
    return assignment


def extract_ffpp(root, split_dir, output, compression="c23", frames=32,
                 crop_size=400, methods=("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"),
                 margin=.3, max_originals=None):
    """Use original-video Haar boxes at identical frame indices for all methods.

    Haar is a lightweight fallback, not the paper's face detector/alignment.
    No center-crop fallback: failed detections are logged and skipped.
    """
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assignment = official_splits(split_dir)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    source_dir = root / "original_sequences" / "youtube" / compression / "videos"
    originals = sorted(source_dir.glob("*.mp4"))
    if not originals:
        raise FileNotFoundError(f"No FF++ original videos found in {source_dir}")
    if max_originals:
        originals = originals[:max_originals]
    records, failures, box_cache = [], [], {}

    def write_frame(frame, box, split, group, video_id, label, index):
        h, w = frame.shape[:2]
        # Fractional boxes remain consistent when encoded dimensions differ.
        x0, y0, x1, y1 = [int(v*s) for v, s in zip(box, [w, h, w, h])]
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            raise ValueError(f"Empty crop in {video_id}")
        relative = Path("crops") / video_id / f"{index:06d}.png"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(destination), crop):
            raise IOError(f"Could not write {destination}")
        records.append(dict(path=relative.as_posix(), label=label, split=split,
                            group_id=group, video_id=video_id, dataset="FF++",
                            frame_index=index, compression=compression))

    for path in tqdm(originals, desc="Original face boxes"):
        source = path.stem
        if source not in assignment:
            continue
        split, group = assignment[source]
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sorted(set(int(i*(count-1)/max(frames-1, 1)) for i in range(frames))) if count else []
        cache = {}
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = cap.read()
            if not success:
                failures.append(dict(video=str(path), frame=index, reason="read_failed"))
                continue
            faces = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                               scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 0:
                failures.append(dict(video=str(path), frame=index, reason="no_face"))
                continue
            x, y, w, h = max(faces, key=lambda box: box[2]*box[3])
            side = max(w, h)*(1+2*margin)
            cx, cy = x+w/2, y+h/2
            fh, fw = frame.shape[:2]
            box = (max(0, cx-side/2)/fw, max(0, cy-side/2)/fh,
                   min(fw, cx+side/2)/fw, min(fh, cy+side/2)/fh)
            cache[index] = box
            write_frame(frame, box, split, group, f"original/{source}", 0, index)
        cap.release()
        box_cache[source] = cache
    for method in methods:
        folder = root / "manipulated_sequences" / method / compression / "videos"
        for path in tqdm(sorted(folder.glob("*.mp4")), desc=method):
            parts = path.stem.split("_")
            if len(parts) != 2 or parts[0] not in box_cache:
                continue
            source, target = parts
            if target not in assignment or assignment[source] != assignment[target]:
                raise ValueError(f"Manipulation crosses split/groups: {path}")
            split, group = assignment[source]
            cap = cv2.VideoCapture(str(path))
            for index, box in box_cache[source].items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame = cap.read()
                if success:
                    write_frame(frame, box, split, group, f"{method}/{path.stem}", 1, index)
                else:
                    failures.append(dict(video=str(path), frame=index, reason="read_failed"))
            cap.release()
    if not records:
        raise RuntimeError("No crops extracted. Inspect video paths and face detections.")
    manifest = output / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    (output / "preprocessing_failures.json").write_text(json.dumps(failures, indent=2))
    (output / "preprocessing_config.json").write_text(json.dumps(dict(
        compression=compression, frames=frames, crop_size=crop_size, methods=list(methods),
        margin=margin, detector="opencv_haar_original_boxes", max_originals=max_originals), indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--compression", default="c23")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--size", type=int, default=400)
    args = p.parse_args()
    print(extract_ffpp(args.root, args.splits, args.output, args.compression, args.frames, args.size))


if __name__ == "__main__":
    main()
