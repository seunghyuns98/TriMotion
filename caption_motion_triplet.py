import argparse
import glob
import json
import os

from natsort import natsorted
from tqdm import tqdm

from caption_generator import (
    PROMPT_LONG,
    PROMPT_SHORT,
    extrinsics_dict_to_pose_tensor,
    generate_caption,
    load_qwen_model,
)


def discover_scenes(dataset_path, mode):
    if mode == "train":
        pattern = os.path.join(dataset_path, "train", "f*", "scene*")
    else:
        pattern = os.path.join(dataset_path, "val", "10*", "scene*")
    return natsorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def load_scene_extrinsics(scene_dir):
    merged_path = os.path.join(scene_dir, "merged_conditions.json")
    extrinsics_path = os.path.join(scene_dir, "cameras", "camera_extrinsics.json")

    if os.path.isfile(merged_path):
        with open(merged_path, "r") as f:
            merged = json.load(f)
        return {
            cam_id: cam_data["extrinsics"]
            for cam_id, cam_data in merged.items()
            if "extrinsics" in cam_data
        }

    if os.path.isfile(extrinsics_path):
        with open(extrinsics_path, "r") as f:
            cam_data = json.load(f)
        cam_ids = sorted(
            {cam for frame_data in cam_data.values() for cam in frame_data}
        )
        return {
            cam_id: {
                frame_key: frame_data[cam_id]
                for frame_key, frame_data in cam_data.items()
                if cam_id in frame_data
            }
            for cam_id in cam_ids
        }

    return None


def process_motion_triplet_dataset(args, model, tokenizer):
    scenes = discover_scenes(args.dataset_path, args.mode)
    if not scenes:
        raise FileNotFoundError(
            f"No scenes found under {args.dataset_path} (mode={args.mode}). "
            "See README Motion Triplet Dataset section for the expected layout."
        )

    caption_types = []
    if args.caption_type in ("long", "both"):
        caption_types.append(("long", PROMPT_LONG, "text_description_long.json"))
    if args.caption_type in ("short", "both"):
        caption_types.append(("short", PROMPT_SHORT, "text_description_short.json"))

    for scene_dir in tqdm(scenes, desc=f"Scenes ({args.mode})"):
        extrinsics_per_cam = load_scene_extrinsics(scene_dir)
        if not extrinsics_per_cam:
            tqdm.write(f"[skip] no extrinsics: {scene_dir}")
            continue

        cameras_dir = os.path.join(scene_dir, "cameras")
        os.makedirs(cameras_dir, exist_ok=True)

        for _, prompt, output_filename in caption_types:
            caption_results = {}
            for cam_id in sorted(extrinsics_per_cam):
                camera_pose = extrinsics_dict_to_pose_tensor(
                    extrinsics_per_cam[cam_id],
                    max_num_frames=args.max_num_frames,
                    frame_interval=args.frame_interval,
                )
                caption_results[cam_id] = generate_caption(
                    model,
                    tokenizer,
                    prompt,
                    camera_pose,
                    max_new_tokens=args.max_new_tokens,
                )

            output_path = os.path.join(cameras_dir, output_filename)
            with open(output_path, "w") as f:
                json.dump(caption_results, f, indent=4, ensure_ascii=False)
            tqdm.write(f"Saved: {output_path}")


def main():
    default_dataset = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "MotionTriplet-Dataset"
    )

    parser = argparse.ArgumentParser(
        description="Generate camera captions for the Motion Triplet Dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=default_dataset,
        help="MotionTriplet-Dataset root (default: ./MotionTriplet-Dataset).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=("train", "val"),
        help="Dataset split.",
    )
    parser.add_argument(
        "--caption-type",
        type=str,
        default="both",
        choices=("long", "short", "both"),
        help="Which caption style to generate.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--max-num-frames",
        type=int,
        default=81,
        help="Maximum frames in extrinsics before subsampling.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=4,
        help="Frame subsampling interval (81 frames -> 21 poses).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens per caption.",
    )
    args = parser.parse_args()

    model, tokenizer = load_qwen_model(args.model_name)
    process_motion_triplet_dataset(args, model, tokenizer)


if __name__ == "__main__":
    main()
