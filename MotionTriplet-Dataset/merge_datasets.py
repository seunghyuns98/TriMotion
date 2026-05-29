"""
1) Merge Motion_Description/ and MultiCamVideo-Dataset/MultiCamVideo-Dataset/
   into MotionTriplet-Dataset/{train,val}/ by moving files.
2) Remove the now-empty source directories.
3) Per scene, combine 4 JSON files into one merged_conditions.json:
     cameras/camera_extrinsics.json      {frame: {cam: matrix}}
     cameras/text_description_long.json  {cam: long_text}
     cameras/text_description_short.json {cam: short_text}
     text/text_description.json          {cam: text}
   Output shape:
     {cam: {"extrinsics": {frame: matrix}, "captions": {"long","short","text"}}}
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ROOT / "Motion_Description",
    ROOT / "MultiCamVideo-Dataset" / "MultiCamVideo-Dataset",
]
SPLITS = ["train", "val"]


def move_sources() -> None:
    for src_root in SOURCES:
        if not src_root.is_dir():
            print(f"[skip] missing source: {src_root}")
            continue
        for split in SPLITS:
            src_split = src_root / split
            if not src_split.is_dir():
                continue
            for src_file in src_split.rglob("*"):
                if not src_file.is_file():
                    continue
                rel = src_file.relative_to(src_split)
                dst = ROOT / split / rel
                if dst.exists():
                    print(f"[exists] {dst}")
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                print(f"[move] {src_file}  ->  {dst}")
                shutil.move(str(src_file), str(dst))

    for src_root in SOURCES:
        if src_root.is_dir():
            print(f"[rmtree] {src_root}")
            shutil.rmtree(src_root)


def merge_scene_conditions(scene_dir: Path) -> None:
    extr_p = scene_dir / "cameras" / "camera_extrinsics.json"
    long_p = scene_dir / "cameras" / "text_description_long.json"
    short_p = scene_dir / "cameras" / "text_description_short.json"
    text_p = scene_dir / "text" / "text_description.json"

    missing = [p.name for p in (extr_p, long_p, short_p, text_p) if not p.is_file()]
    if missing:
        print(f"[skip] {scene_dir}: missing {missing}")
        return

    extr = json.load(open(extr_p))
    longs = json.load(open(long_p))
    shorts = json.load(open(short_p))
    texts = json.load(open(text_p))

    cams = sorted(set(longs) | set(shorts) | set(texts) |
                  {c for fd in extr.values() for c in fd})

    merged = {}
    for cam in cams:
        merged[cam] = {
            "extrinsics": {f: fd[cam] for f, fd in extr.items() if cam in fd},
            "captions": {
                "long": longs.get(cam, ""),
                "short": shorts.get(cam, ""),
                "text": texts.get(cam, ""),
            },
        }

    out_p = scene_dir / "merged_conditions.json"
    with open(out_p, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"[ok] {out_p}")


def merge_all_conditions() -> None:
    for split in SPLITS:
        split_dir = ROOT / split
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(split_dir.glob("*/scene*")):
            if scene_dir.is_dir():
                merge_scene_conditions(scene_dir)


def main() -> None:
    # move_sources()
    merge_all_conditions()


if __name__ == "__main__":
    main()
